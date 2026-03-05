import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from sc.clustering.dataloader import AuxSpectraDataset
from sc.report.calibration import load_calibrator, predict_d_from_z


def _nan_rmse(y, yhat):
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    m = ~(np.isnan(y) | np.isnan(yhat))
    if not np.any(m):
        return np.nan
    e2 = (yhat[m] - y[m]) ** 2
    return float(np.sqrt(np.mean(e2)))


def _nan_mae(y, yhat):
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    m = ~(np.isnan(y) | np.isnan(yhat))
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs(yhat[m] - y[m])))


def _nan_r2(y, yhat):
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    m = ~(np.isnan(y) | np.isnan(yhat))
    if np.sum(m) < 2:
        return np.nan
    try:
        return float(r2_score(y[m], yhat[m]))
    except Exception:
        return np.nan


def _nan_spearman(y, yhat):
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    m = ~(np.isnan(y) | np.isnan(yhat))
    if np.sum(m) < 2:
        return np.nan
    c = spearmanr(y[m], yhat[m]).correlation
    return float(c) if c is not None else np.nan


def _infer_aux_names_from_csv(csv_path, n_aux):
    """
    Try to infer AUX column names from the CSV header.
    Preference: columns starting with 'AUX_' (in file order).
    Returns list[str] or None if cannot infer.
    """
    try:
        df0 = pd.read_csv(csv_path, nrows=1)
    except Exception:
        return None

    aux_cols = [c for c in df0.columns if str(c).startswith("AUX_")]
    if len(aux_cols) >= n_aux:
        return aux_cols[:n_aux]
    return None


def _as_id_cols(atom_index):
    """
    AuxSpectraDataset uses df.index.to_list() where index_col=[0,1] -> tuples.
    Return two columns if tuple-like, else one column.

    We name them MPID and SITE (instead of id0/id1).
    """
    if atom_index is None or len(atom_index) == 0:
        return {}, ["MPID"]

    first = atom_index[0]
    if isinstance(first, tuple) and len(first) == 2:
        col0 = [str(x[0]) for x in atom_index]
        col1 = [str(x[1]) for x in atom_index]
        return {"MPID": col0, "SITE": col1}, ["MPID", "SITE"]
    else:
        return {"MPID": [str(x) for x in atom_index]}, ["MPID"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pt", type=str, required=True, help="Path to final.pt or best.pt")
    ap.add_argument("--calib_pkl", type=str, required=True, help="Path to saved z->d calibrator .pkl")
    ap.add_argument("--test_csv", type=str, required=True, help="Held-out test CSV")
    ap.add_argument("--n_aux", type=int, required=True, help="Number of descriptors (AUX columns)")
    ap.add_argument("--aux_names", type=str, default=None, help="Optional comma-separated names for aux columns")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--random_seed", type=int, default=0)

    ap.add_argument("--out_prefix", type=str, default="predictions", help="Output prefix (no extension)")
    ap.add_argument("--eps", type=float, default=1e-12, help="Epsilon for normalized errors")

    args = ap.parse_args()
    device = torch.device(args.device)

    # Descriptor names (prefer: user-provided -> inferred from CSV header -> fallback AUX_1..)
    if args.aux_names is not None:
        aux_names = [s.strip() for s in args.aux_names.split(",") if s.strip()]
        if len(aux_names) != args.n_aux:
            raise ValueError(f"--aux_names must have length n_aux={args.n_aux}, got {len(aux_names)}")
    else:
        inferred = _infer_aux_names_from_csv(args.test_csv, args.n_aux)
        aux_names = inferred if inferred is not None else [f"AUX_{i+1}" for i in range(args.n_aux)]

    # Load test set: treat whole file as a single "train" split.
    test_ds = AuxSpectraDataset(
        args.test_csv,
        split_portion="train",
        train_val_test_ratios=(1.0, 0.0, 0.0),
        n_aux=args.n_aux,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
    )
    if test_ds.aux is None:
        raise ValueError("Test dataset has no aux columns (n_aux=0 or missing AUX_ columns).")

    # Load model + calibrator
    model = torch.load(args.model_pt, map_location=device, weights_only=False)
    encoder = model["Encoder"].to(device)
    encoder.eval()

    calibrator = load_calibrator(args.calib_pkl)

    # Encode S -> z -> d_hat
    S = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = encoder(S).detach().cpu().numpy()

    d_hat = predict_d_from_z(calibrator, z)
    d_true = np.asarray(test_ds.aux, dtype=float)

    if d_hat.shape != d_true.shape:
        raise ValueError(f"Shape mismatch: d_hat {d_hat.shape} vs d_true {d_true.shape}")

    # Baseline: constant predictor per descriptor.
    baseline_mean = calibrator.get("baseline_mean", None)
    if baseline_mean is None:
        baseline_mean = calibrator.get("y_mean", None)

    if baseline_mean is None:
        print("WARNING: calibrator has no baseline_mean/y_mean; using test-set mean as baseline (leaky).")
        baseline_mean = np.nanmean(d_true, axis=0)
    baseline_mean = np.asarray(baseline_mean, dtype=float).reshape(-1)
    if baseline_mean.shape[0] != args.n_aux:
        raise ValueError(f"Baseline mean length {baseline_mean.shape[0]} != n_aux={args.n_aux}")

    d_base = np.broadcast_to(baseline_mean.reshape(1, -1), d_true.shape)

    # --- Per-sample CSV in LONG format (one row per (sample, descriptor)) ---
    id_cols_dict, id_colnames = _as_id_cols(test_ds.atom_index)
    N = d_true.shape[0]
    rows = []

    for j in range(N):
        for i in range(args.n_aux):
            y = d_true[j, i]
            yhat = d_hat[j, i]
            yb = d_base[j, i]

            if np.isnan(y) or np.isnan(yhat):
                rmse_1 = np.nan
            else:
                rmse_1 = float(np.sqrt((yhat - y) ** 2))

            if np.isnan(y) or np.isnan(yb):
                rmse_base_1 = np.nan
            else:
                rmse_base_1 = float(np.sqrt((yb - y) ** 2))

            nrmse_1 = (
                rmse_1 / (rmse_base_1 + args.eps)
                if (rmse_1 == rmse_1) and (rmse_base_1 == rmse_base_1)
                else np.nan
            )

            # Put MPID/SITE first
            r = {}
            for k in id_colnames:
                r[k] = id_cols_dict[k][j]

            r.update({
                "sample_index": j,
                "descriptor_index": i,
                "descriptor_name": aux_names[i],
                "true": y,
                "pred": yhat,
                "baseline_pred": yb,
                "rmse": rmse_1,
                "baseline_rmse": rmse_base_1,
                "normalized_rmse_vs_baseline": nrmse_1,
            })
            rows.append(r)

    df_per_sample = pd.DataFrame(rows)

    # Ensure column order: MPID, SITE, then the rest
    desired_cols = (
        id_colnames
        + [
            "sample_index",
            "descriptor_index",
            "descriptor_name",
            "true",
            "pred",
            "baseline_pred",
            "rmse",
            "baseline_rmse",
            "normalized_rmse_vs_baseline",
        ]
    )
    df_per_sample = df_per_sample[desired_cols]

    out_per_sample = f"{args.out_prefix}_per_sample.csv"
    df_per_sample.to_csv(out_per_sample, index=False)

    # --- Summary over entire test set ---
    summary = []
    for i in range(args.n_aux):
        y = d_true[:, i]
        yhat = d_hat[:, i]
        yb = d_base[:, i]

        rmse = _nan_rmse(y, yhat)
        mae = _nan_mae(y, yhat)
        r2 = _nan_r2(y, yhat)
        sp = _nan_spearman(y, yhat)

        rmse_base = _nan_rmse(y, yb)
        mae_base = _nan_mae(y, yb)

        nrmse = rmse / (rmse_base + args.eps) if (rmse == rmse) and (rmse_base == rmse_base) else np.nan
        nmae = mae / (mae_base + args.eps) if (mae == mae) and (mae_base == mae_base) else np.nan

        summary.append({
            "descriptor_index": i,
            "descriptor_name": aux_names[i],
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman": sp,
            "baseline_rmse": rmse_base,
            "baseline_mae": mae_base,
            "normalized_rmse_vs_baseline": nrmse,
            "normalized_mae_vs_baseline": nmae,
            "baseline_mean_used": float(baseline_mean[i]),
        })

    df_summary = pd.DataFrame(summary)

    # Overall aggregates:
    m_all = ~(np.isnan(d_true) | np.isnan(d_hat))
    overall_rmse = float(np.sqrt(np.mean((d_hat[m_all] - d_true[m_all]) ** 2))) if np.any(m_all) else np.nan
    overall_baseline_rmse = float(np.sqrt(np.mean((d_base[m_all] - d_true[m_all]) ** 2))) if np.any(m_all) else np.nan
    overall_nrmse = (
        overall_rmse / (overall_baseline_rmse + args.eps)
        if (overall_rmse == overall_rmse) and (overall_baseline_rmse == overall_baseline_rmse)
        else np.nan
    )

    overall_mean_nrmse = float(np.nanmean(df_summary["normalized_rmse_vs_baseline"].to_numpy()))

    overall_row = {
        "descriptor_index": -1,
        "descriptor_name": "OVERALL",
        "rmse": overall_rmse,
        "mae": float(np.nan),
        "r2": float(np.nan),
        "spearman": float(np.nan),
        "baseline_rmse": overall_baseline_rmse,
        "baseline_mae": float(np.nan),
        "normalized_rmse_vs_baseline": overall_nrmse,
        "normalized_mae_vs_baseline": float(np.nan),
        "baseline_mean_used": float(np.nan),
        "overall_mean_normalized_rmse_vs_baseline": overall_mean_nrmse,
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([overall_row])], ignore_index=True)

    out_summary = f"{args.out_prefix}_summary.csv"
    df_summary.to_csv(out_summary, index=False)

    out_json = f"{args.out_prefix}_summary.json"
    payload = {
        "model_pt": os.path.abspath(args.model_pt),
        "calib_pkl": os.path.abspath(args.calib_pkl),
        "test_csv": os.path.abspath(args.test_csv),
        "n_aux": int(args.n_aux),
        "overall_rmse": overall_rmse,
        "overall_baseline_rmse": overall_baseline_rmse,
        "overall_normalized_rmse_vs_baseline": overall_nrmse,
        "overall_mean_normalized_rmse_vs_baseline": overall_mean_nrmse,
        "per_descriptor": summary,
        "outputs": {
            "per_sample_csv": os.path.abspath(out_per_sample),
            "summary_csv": os.path.abspath(out_summary),
        },
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("Wrote:")
    print(" ", out_per_sample)
    print(" ", out_summary)
    print(" ", out_json)
    print("Overall RMSE:", overall_rmse)
    print("Overall baseline RMSE:", overall_baseline_rmse)
    print("Overall normalized RMSE vs baseline:", overall_nrmse)
    print("Overall mean(per-descriptor normalized RMSE):", overall_mean_nrmse)


if __name__ == "__main__":
    main()
