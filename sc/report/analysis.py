import math
import os
import itertools
import torch
import pickle
import numpy as np
from numpy.polynomial import Polynomial
from scipy import stats
from scipy.stats import spearmanr, shapiro
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def create_plotly_colormap(n_colors):
    '''
    Xiaohui's implementation of getting spectra color map.
    '''
    plotly3_hex_strings = px.colors.sequential.Plotly3
    rgb_values = np.array([[int(f"0x{c_hex[i:i+2]}", 16) for i in range(1, 7, 2)] for c_hex in plotly3_hex_strings])
    x0 = np.linspace(1, n_colors, rgb_values.shape[0])
    x1 = np.linspace(1, n_colors, n_colors)
    target_rgb_values = np.stack([interp1d(x0, rgb_values[:, i], kind='cubic')(x1) for i in range(3)]).T.round().astype('int')
    target_rgb_strings = ["#" + "".join([f'{ch:02x}' for ch in rgb]) for rgb in target_rgb_values]
    return target_rgb_strings


def plot_spectra_variation(
    decoder, istyle,
    n_spec=50,
    n_sampling=1000,
    true_range=True,
    styles=None,
    amplitude= [0,0],
    device=torch.device("cpu"),
    ax=None,
    energy_grid=None,
    colors=None,
    plot_residual=False,
    **kwargs
):
    """
    Spectra variation plot by varying one of the styles.

    Parameters
    ----------
    istyle : int
        The column index of `styles` for which the variation is plotted.
    true_range : bool
        If True, sample from the 5th percentile to 95th percentile of a style, instead of
        [-amplitude, +amplitude].
    n_sampling : int
        The number of random "styles" sampled. If zero, then set other styles to zero.
    amplitude : float
        The range from which styles are sampled. Effective only if `true_range` is False.
    style : array_like
        2-D array of complete styles. Effective and can't be None if `true_range` evaluates
        True.
    plot_residual : bool
        Whether to plot the difference between two extrema instead of all variations.
    """
    decoder.eval()

    ### !!! 
    #true_range = False 
    if true_range:
        if styles is None:
            raise ValueError("`styles` must be provided when `true_range=True`.")
        left, right = np.percentile(styles[:, istyle], [5, 95])
    else:
        left, right = amplitude[0], amplitude[1]
        #left, right = -0.4, 1.5

    # infer output length (avoid hardcoding 2400)
    with torch.no_grad():
        dummy = torch.zeros((1, decoder.nstyle), device=device)
        out_dim = int(decoder(dummy).reshape(1, -1).shape[-1])

    if n_sampling == 0:
        c = np.linspace(left, right, n_spec)
        c2 = np.stack(
            [np.zeros_like(c)] * istyle + [c] + [np.zeros_like(c)] * (decoder.nstyle - istyle - 1),
            axis=1
        )
        con_c = torch.tensor(c2, dtype=torch.float, requires_grad=False, device=device)
        spec_out = decoder(con_c).reshape(n_spec, -1).clone().cpu().detach().numpy()
        style_variation = c
    else:
        con_c = torch.randn([n_spec, n_sampling, decoder.nstyle], device=device)
        style_variation = torch.linspace(left, right, n_spec, device=device)
        con_c[..., istyle] = style_variation[:, np.newaxis]
        con_c = con_c.reshape(n_spec * n_sampling, decoder.nstyle)

        spec_out = decoder(con_c).reshape(n_spec, n_sampling, out_dim)
        spec_out = spec_out.mean(axis=1).cpu().detach().numpy()

    if ax is not None:
        if colors is None:
            colors = create_plotly_colormap(n_spec)
        assert len(colors) == n_spec

        if plot_residual:
            if energy_grid is None:
                ax.plot(spec_out[-1] - spec_out[0], **kwargs)
            else:
                ax.plot(energy_grid, spec_out[-1] - spec_out[0], **kwargs)
            ax.set_ylim([-0.5, 0.5])
        else:
            for spec, color in zip(spec_out, colors):
                if energy_grid is None:
                    ax.plot(spec, c=color, **kwargs)
                else:
                    ax.plot(energy_grid, spec, c=color, **kwargs)

        ax.set_title(f"Style {istyle + 1} varying from {left:.2f} to {right:.2f}", y=1)

    return style_variation, spec_out


def evaluate_all_models(
    model_path, test_ds,
    device=torch.device('cpu')
):
    '''
    Sort models according to multi metrics, in descending order of goodness.
    '''
    result = {}
    for job in os.listdir(model_path):
        if job.startswith("job_"):
            best_path = os.path.join(model_path, job, "best.pt")

            final_path = os.path.join(model_path, job, "final.pt")

            load_path = best_path if os.path.exists(best_path) else final_path

            if not os.path.exists(final_path):
                continue
            
            model = torch.load(
                os.path.join(model_path, job, load_path),
                map_location=device,
                weights_only=False
            )
            result[job] = evaluate_model(test_ds, model, device=device)

    return result


def load_evaluations(evaluation_path="./report_model_evaluations.pkl"):
    with open(evaluation_path, 'rb') as f:
        result = pickle.load(f)
    return result


def sort_all_models(
    result_dict,
    sort_score=None,
    plot_score=False,
    ascending=True,
    top_n=None,
    true_value=True,
    descriptor_names=None
):
    """
    Given the input result dict, calculate (and plot) the score matrix.
    Update the "rank" attribute and return the updated result dict.
    Add key "score" to the result.

    Notes:
    - No longer hardcodes particular physics descriptors.
    - Uses whatever descriptor correlation metrics are present in result_dict[job]["Style-descriptor Corr"].
    """

    def _metric_from_corr_item(item):
        # Prefer F1 if present (discrete descriptor), else Spearman, else Linear R2, else Quadratic R2.
        if item is None:
            return None
        if isinstance(item, dict):
            if "F1 score" in item and item["F1 score"] is not None:
                return float(item["F1 score"])
            if "Spearman" in item and item["Spearman"] is not None:
                return float(item["Spearman"])
            if "Linear" in item and isinstance(item["Linear"], dict) and item["Linear"].get("R2", None) is not None:
                return float(item["Linear"]["R2"])
            if "Quadratic" in item and isinstance(item["Quadratic"], dict) and item["Quadratic"].get("R2", None) is not None:
                return float(item["Quadratic"]["R2"])
        return None

    # gather all descriptor indices present across jobs
    desc_indices = set()
    for _, res in result_dict.items():
        corr = res.get("Style-descriptor Corr", {})
        if isinstance(corr, dict):
            desc_indices |= set(corr.keys())
    desc_indices = sorted([i for i in desc_indices if isinstance(i, (int, np.integer))])

    # build dynamic score_names
    score_names = ["Inter-style Corr", "Reconstruction Err (MAE mean)"]
    for i in desc_indices:
        if descriptor_names is not None and i < len(descriptor_names):
            score_names.append(f"Style_{i + 1} - {descriptor_names[i]}")
        else:
            score_names.append(f"Style_{i + 1} - Descriptor_{i + 1}")

    scores = []
    jobs = []

    for job, res in result_dict.items():
        jobs.append(job)

        inter = res.get("Inter-style Corr", None)
        rec = res.get("Reconstruct Err", (None, None))
        rec_mean = None
        if isinstance(rec, (list, tuple)) and len(rec) > 0:
            rec_mean = rec[0]

        row = [
            float(inter) if inter is not None else 0.0,
            float(rec_mean) if rec_mean is not None else 0.0,
        ]

        corr = res.get("Style-descriptor Corr", {})
        for i in desc_indices:
            v = _metric_from_corr_item(corr.get(i, None))
            row.append(float(v) if v is not None and np.isfinite(v) else 0.0)

        scores.append(row)

    jobs = np.array(jobs)
    scores = np.array(scores, dtype=float)

    # z-score normalize safely (avoid runtime warnings)
    col_mean = np.nanmean(scores, axis=0)
    col_std = np.nanstd(scores, axis=0)
    z_scores = np.zeros_like(scores, dtype=float)
    np.divide(
        (scores - col_mean),
        col_std,
        out=z_scores,
        where=(col_std != 0)
    )
    z_scores[~np.isfinite(z_scores)] = 0.0

    mu_std = np.stack((col_mean, col_std), axis=1)

    # sort scores
    if callable(sort_score):
        final_score = sort_score(z_scores)
    elif isinstance(sort_score, int) and sort_score >= 0 and sort_score < scores.shape[1]:
        final_score = scores[:, sort_score]
    else:
        final_score = np.arange(len(scores))

    rank = np.argsort(final_score)
    if (sort_score is not None) and (not ascending):
        rank = rank[::-1]

    ranked_scores = scores[rank]
    ranked_final_scores = np.array(final_score)[rank]
    ranked_jobs = jobs[rank]
    ranked_z_scores = z_scores[rank]

    for i, (job, score) in enumerate(zip(ranked_jobs, ranked_final_scores)):
        result_dict[job]['Rank'] = i
        result_dict[job]['Score'] = round(float(score), 4)

    fig = None
    if plot_score:
        if top_n is None or top_n > len(ranked_z_scores):
            top_n = len(ranked_z_scores)

        fig, ax = plt.subplots(figsize=(top_n, scores.shape[1]))
        ax.autoscale(enable=True)
        sns.heatmap(
            ranked_z_scores[:top_n].T,
            vmin=-3, vmax=3,
            cmap='Blues', cbar=True,
            annot=ranked_z_scores[:top_n].T if not true_value else ranked_scores[:top_n].T,
            ax=ax,
            yticklabels=[
                f"{name}\n{ms[0]:.3f}+-{ms[1]:.3f}" for name, ms in zip(score_names, mu_std)
            ],
            xticklabels=[
                f"{ranked_jobs[i]}: {ranked_final_scores[i]:.2f}" for i in range(top_n)
            ]
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', va='bottom')
        ax.tick_params(labelbottom=False, labeltop=True, axis='both', length=0, labelsize=15)

    return result_dict, ranked_jobs, fig


def get_confusion_matrix(descriptor, style_value, ax=None, class_labels=None):
    """
    Confusion matrix + best threshold(s) for a discrete descriptor.

    - Maps arbitrary discrete labels to indices 0..K-1.
    - Orders classes by median style value and uses midpoints as thresholds.
    """
    result = {
        "F1 score": None,
        "Classes": None,
        "Thresholds": None,
    }

    descriptor = np.asarray(descriptor)
    style_value = np.asarray(style_value)

    # remove NaNs
    mask = ~(np.isnan(descriptor) | np.isnan(style_value))
    descriptor = descriptor[mask]
    style_value = style_value[mask]
    if descriptor.size == 0:
        return None

    # determine classes (only those present)
    if class_labels is None:
        classes = np.unique(descriptor)
        try:
            classes = np.sort(classes)
        except Exception:
            pass
    else:
        classes = np.asarray(class_labels)

    # ---- FIX: define y_true BEFORE using it ----
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = np.array([class_to_idx.get(v, -1) for v in descriptor], dtype=int)
    valid = (y_true >= 0)
    y_true = y_true[valid]
    style_value = style_value[valid]

    if y_true.size == 0:
        return None

    k = len(classes)
    result["Classes"] = [str(c) for c in classes]

    # order classes by median style value
    med = np.array([np.median(style_value[y_true == i]) for i in range(k)], dtype=float)
    order = np.argsort(med)

    # remap y_true into ordered-class index space
    new_idx = np.zeros(k, dtype=int)
    new_idx[order] = np.arange(k)
    y_true_ord = new_idx[y_true]

    classes_ord = classes[order]
    med_ord = med[order]

    # thresholds between adjacent medians
    thresholds = [(med_ord[i] + med_ord[i + 1]) / 2.0 for i in range(k - 1)]
    y_pred = np.digitize(style_value, thresholds)  # 0..k-1

    cm = confusion_matrix(y_true_ord, y_pred, labels=list(range(k)))
    w_f1 = f1_score(y_true_ord, y_pred, average="weighted", zero_division=0)

    result["F1 score"] = round(float(w_f1), 4)
    result["Classes"] = [str(c) for c in classes_ord]
    result["Thresholds"] = [round(float(t), 4) for t in thresholds]

    if ax is not None:
        ax[0].cla()
        ax[0].hist([style_value[y_true_ord == i] for i in range(k)],
                   bins=40, stacked=True, alpha=0.7,
                   label=[str(c) for c in classes_ord])
        for th in thresholds:
            ax[0].axvline(th, c="k", alpha=0.6)
        ax[0].legend(fontsize=9)
        ax[0].set_title(f"Discrete separation (K={k})", fontsize=10)

        sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", cbar=False, ax=ax[1],
                    xticklabels=[str(c) for c in classes_ord],
                    yticklabels=[str(c) for c in classes_ord])
        ax[1].set_title(f"F1 Score = {w_f1:.1%}", fontsize=12)
        ax[1].set_xlabel("Pred")
        ax[1].set_ylabel("True")

        colors = np.array(sns.color_palette("bright", k))
        test_colors = colors[y_true_ord]
        test_colors = np.array([mpl.colors.colorConverter.to_rgba(c, alpha=0.6) for c in test_colors])
        random_style = np.random.uniform(style_value.min(), style_value.max(), len(style_value))
        ax[2].scatter(style_value, random_style, s=10.0, color=test_colors, alpha=0.8)
        for th in thresholds:
            ax[2].axvline(th, c="gray", alpha=0.8)
        ax[2].set_xlabel("Style (discrete corr)")
        ax[2].set_ylabel("Random")

    return result


def get_max_inter_style_correlation(styles):
    """
    Maximum absolute Spearman correlation between ANY pair of style dimensions.
    Robust to any number of styles, and avoids hardcoded assumptions.
    """
    styles = np.asarray(styles)
    if styles.ndim != 2 or styles.shape[1] < 2:
        return 0.0

    corr_list = []
    for i, j in itertools.combinations(range(styles.shape[1]), 2):
        a = styles[:, i]
        b = styles[:, j]
        # if constant vectors, spearman returns nan
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            corr = 0.0
        else:
            corr = spearmanr(a, b).correlation
            corr = 0.0 if (corr is None or not np.isfinite(corr)) else float(corr)
        corr_list.append(abs(corr))

    return round(float(max(corr_list) if corr_list else 0.0), 4)


def get_descriptor_style_correlation(
    style,
    descriptor,
    ax=None,
    choice=["R2", "Spearman"],
    fit=True
):
    """
    Calculate relations between styles and descriptors including R^2, Spearman, Polynomial/Linear fitting etc.
    If axis is given, scatter plot of given descriptor and style is also plotted.
    """
    style = np.asarray(style)
    descriptor = np.asarray(descriptor)

    # sort by style for nicer plotting/fit
    sorted_index = np.argsort(style)
    style = style[sorted_index]
    descriptor = descriptor[sorted_index]

    # mask out NaNs
    mask_nan = ~(np.isnan(descriptor) | np.isnan(style))
    style = style[mask_nan]
    descriptor = descriptor[mask_nan]

    accuracy = {
        "Spearman": None,
        "Linear": {
            "slope": None,
            "intercept": None,
            "R2": None
        },
        "Quadratic": {
            "Parameters": [None, None, None],
            "residue": None,
            "R2": None
        }
    }

    if style.size < 2 or descriptor.size < 2:
        return accuracy

    if np.nanstd(style) == 0 or np.nanstd(descriptor) == 0:
        # undefined correlations/fit
        return accuracy

    fitted_value = None

    if "R2" in choice:
        result = stats.linregress(style, descriptor)
        r2 = float(result.rvalue ** 2) if np.isfinite(result.rvalue) else None
        accuracy["Linear"]["R2"] = np.round(r2, 4).tolist() if r2 is not None else None
        accuracy["Linear"]["intercept"] = np.round(float(result.intercept), 4).tolist()
        accuracy["Linear"]["slope"] = np.round(float(result.slope), 4).tolist()
        fitted_value = result.intercept + style * result.slope

    if "Spearman" in choice:
        sm = spearmanr(style, descriptor).correlation
        sm = None if (sm is None or not np.isfinite(sm)) else float(sm)
        accuracy["Spearman"] = np.round(sm, 4).tolist() if sm is not None else None

    if "Quadratic" in choice:
        p, info = Polynomial.fit(style, descriptor, 2, full=True)
        accuracy["Quadratic"]["Parameters"] = np.round(p.convert().coef, 4).tolist()

        residuals = np.asarray(info[0])  # may be empty or shape (1,)
        if residuals.size > 0 and len(style) > 0:
            residue = float(residuals.ravel()[0]) / len(style)
        else:
            residue = None

        accuracy["Quadratic"]["residue"] = None if residue is None else round(residue, 4)
        fitted_value = p(style)

        # q_r2 safety
        if np.nanstd(fitted_value) == 0 or np.nanstd(descriptor) == 0:
            accuracy["Quadratic"]["R2"] = None
        else:
            q_r = stats.linregress(fitted_value, descriptor).rvalue
            accuracy["Quadratic"]["R2"] = None if not np.isfinite(q_r) else round(float(q_r**2), 4)
    

    if ax is not None:
        ax.scatter(style, descriptor, s=10.0, c='blue', edgecolors='none', alpha=0.8)
        if fit and fitted_value is not None:
            ax.plot(style, fitted_value, lw=2, c='black', alpha=0.5)

    return accuracy


def evaluate_model(
    test_ds,
    model,
    reconstruct=True,
    accuracy=True,
    style=True,
    device=torch.device('cpu'),
    discrete_descriptor_max_classes=3
):
    '''
    calculate reconstruction error for a given model, or accuracy.

    - No longer hardcodes "descriptor 1 is CN".
    - Automatically treats a descriptor as "discrete" iff:
        * it is integer-like AND
        * it has <= `discrete_descriptor_max_classes` unique values (after NaN removal).
    '''
    descriptors = test_ds.aux

    result = {
        "Style-descriptor Corr": {},
        "Input": None,
        "Output": None,
        "Reconstruct Err": (None, None),
        "Inter-style Corr": None
    }

    encoder = model['Encoder']
    decoder = model['Decoder']
    encoder.eval()
    decoder.eval()

    spec_in = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    styles_t = encoder(spec_in)
    result["Input"] = spec_in.detach().cpu().numpy()

    if reconstruct:
        spec_out = decoder(styles_t).clone().detach().cpu().numpy()
        mae_list = [mean_absolute_error(s1, s2) for s1, s2 in zip(result["Input"], spec_out)]
        result["Reconstruct Err"] = [
            round(float(np.mean(mae_list)), 4),
            round(float(np.std(mae_list)), 4)
        ]
        result["Output"] = spec_out

    styles = styles_t.clone().detach().cpu().numpy()

    if accuracy:
        n_desc = descriptors.shape[1] if (hasattr(descriptors, "shape") and descriptors.ndim == 2) else 0
        n_style = styles.shape[1] if (hasattr(styles, "shape") and styles.ndim == 2) else 0
        n = min(n_desc, n_style)

        for i in range(n):
            d = descriptors[:, i]
            s = styles[:, i]

            # decide discrete vs continuous
            d_nonan = d[~np.isnan(d)] if np.issubdtype(d.dtype, np.floating) else d
            # integer-like check: float values very close to integers
            if d_nonan.size > 0 and np.issubdtype(d_nonan.dtype, np.floating):
                int_like = np.all(np.isclose(d_nonan, np.round(d_nonan), atol=1e-6))
            else:
                int_like = np.issubdtype(d_nonan.dtype, np.integer)

            unique_count = len(np.unique(d_nonan)) if d_nonan.size > 0 else 0

            if int_like and (1 < unique_count <= discrete_descriptor_max_classes):
                result["Style-descriptor Corr"][i] = get_confusion_matrix(
                    np.round(d).astype(int),
                    s,
                    ax=None
                )
            else:
                result["Style-descriptor Corr"][i] = get_descriptor_style_correlation(
                    s, d, ax=None, choice=["R2", "Spearman", "Quadratic"]
)
                

    if style:
        result["Inter-style Corr"] = get_max_inter_style_correlation(styles)

    return result


def qqplot_normal(x, ax=None, grid=True):
    """
    Examine the "normality" of a distribution using qqplot.
    Return the Shapiro statistic that represent the similarity of `x` to normality.
    """
    x = np.asarray(x).ravel()
    if x.size < 3:
        return np.nan

    # standardize input data
    if np.nanstd(x) == 0:
        return np.nan

    x_std = (x - np.nanmean(x)) / np.nanstd(x)
    z_score = np.sort(x_std)

    # sample from standard normal distribution and calculate quantiles
    normal = np.random.randn(len(z_score))
    q_normal = np.quantile(normal, np.linspace(0, 1, len(z_score)))

    shapiro_statistic = shapiro(z_score).statistic

    if ax is not None:
        ax.plot(q_normal, z_score, ls='', marker='.', color='k')
        ax.plot([q_normal.min(), q_normal.max()], [q_normal.min(), q_normal.max()],
                color='k', alpha=0.5)
        ax.grid(grid)

    return shapiro_statistic
