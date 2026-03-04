import pickle
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

def _mask_xy(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    m = ~(np.isnan(x) | np.isnan(y))
    return x[m], y[m]

class Linear1D:
    def __init__(self, slope=0.0, intercept=0.0):
        self.slope = float(slope)
        self.intercept = float(intercept)

    def predict(self, x):
        x = np.asarray(x).reshape(-1)
        return self.intercept + self.slope * x

def fit_z_to_d_calibrator(
    z,                  # (N, nstyle)
    d,                  # (N, n_aux)
    method="Isotonic",  # "Isotonic" or "Linear"
    use_first_n_aux=True,
):
    z = np.asarray(z)
    d = np.asarray(d)
    n_aux = d.shape[1]
    models = []

    # NEW: baseline mean per descriptor (computed on the data used to fit calibrator)
    baseline_mean = np.nanmean(d, axis=0)  # shape (n_aux,)

    for i in range(n_aux):
        x = z[:, i] if use_first_n_aux else z[:, i]
        y = d[:, i]
        x, y = _mask_xy(x, y)

        if len(x) < 2 or np.allclose(x, x[0]):
            # degenerate: predict constant
            if method == "Linear":
                models.append(Linear1D(0.0, float(np.nanmean(y)) if len(y) else 0.0))
            else:
                ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
                ir.fit([0.0, 1.0], [float(np.nanmean(y)) if len(y) else 0.0]*2)
                models.append(ir)
            continue

        if method.lower() == "linear":
            slope, intercept = np.polyfit(x, y, deg=1)
            models.append(Linear1D(slope=slope, intercept=intercept))

        elif method.lower() == "isotonic":
            ir = IsotonicRegression(increasing="auto", out_of_bounds="clip")
            ir.fit(x, y)
            models.append(ir)

        else:
            raise ValueError(f"Unknown method: {method}")

    return {
        "method": method,
        "n_aux": int(n_aux),
        "models": models,

        # NEW: save baseline mean inside calibrator so eval doesn't use test mean
        "baseline_mean": baseline_mean,

        # OPTIONAL: keep old key name too for backward compatibility
        "y_mean": baseline_mean,
    }

def predict_d_from_z(calibrator, z):
    z = np.asarray(z)
    n_aux = calibrator["n_aux"]
    d_hat = np.zeros((z.shape[0], n_aux), dtype=float)
    for i, mdl in enumerate(calibrator["models"]):
        x = z[:, i]
        if hasattr(mdl, "predict"):
            d_hat[:, i] = mdl.predict(x)
        else:
            d_hat[:, i] = np.nan
    return d_hat

def save_calibrator(calibrator, path):
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)

def load_calibrator(path):
    with open(path, "rb") as f:
        return pickle.load(f)
