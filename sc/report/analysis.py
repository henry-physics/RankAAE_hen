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
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression

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
    target_rgb_strings = ["#"+"".join([f'{ch:02x}' for ch in rgb]) for rgb in target_rgb_values]
    return target_rgb_strings


def plot_spectra_variation(
    decoder, istyle,
    n_spec = 50, 
    n_sampling = 1000, 
    true_range = True,
    styles = None,
    amplitude = 2,
    device = torch.device("cpu"),
    ax = None,
    energy_grid = None,
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
        2-D array of complete styles. Effective and can't be None if `true_range` evaluates.
        True. The `istyle`th column 
    plot_residual : bool
        Weather to plot out the difference between two extrema instead of all variations.
    """

    decoder.eval()
    left, right = np.percentile(styles[:, istyle], [5, 95])
    if n_sampling == 0: 
        c = np.linspace(left, right, n_spec)
        c2 = np.stack([np.zeros_like(c)] * istyle + [c] + [np.zeros_like(c)] * (decoder.nstyle - istyle - 1), axis=1)
        con_c = torch.tensor(c2, dtype=torch.float, requires_grad=False, device=device)
        spec_out = decoder(con_c).reshape(n_spec, -1).clone().cpu().detach().numpy()
        style_variation = c
    else:
        # Create a 3-D array whose x,y,z dimensions correspond to: style variation, n_sampling, 
        # and number of styles. 
        con_c = torch.randn([n_spec, n_sampling, decoder.nstyle], device=device)
        assert len(styles.shape) == 2 # styles must be a 2-D array
        style_variation = torch.linspace(left, right, n_spec, device=device)
        # Assign the "layer" to be duplicates of `style_variation`
        con_c[..., istyle] = style_variation[:, np.newaxis]
        con_c = con_c.reshape(n_spec * n_sampling, decoder.nstyle)
        spec_out = decoder(con_c).reshape(n_spec, n_sampling, -1)
        # Average along the `n_sampling` dimsion.
        spec_out = spec_out.mean(axis = 1).cpu().detach().numpy()
    
    if ax is not None:
        if colors is None:
            colors = create_plotly_colormap(n_spec)
        assert len(colors) == n_spec
        for spec, color in zip(spec_out, colors):
            if energy_grid is None: # whether use energy as x-axis unit
                ax.plot(spec, c=color, **kwargs)
            elif plot_residual:  # whether plot raw variation or residual between extrema
                ax.plot(energy_grid, spec_out[-1]-spec_out[0], **kwargs)
                ax.set_ylim([-0.5, 0.5])
                break
            else:
                ax.plot(energy_grid, spec, c=color, **kwargs)
        ax.set_title(f"Style {istyle+1} varying from {left:.2f} to {right:.2f}", y=1)

    return style_variation, spec_out

def evaluate_all_models(
    model_path, test_ds,
    device=torch.device('cpu'),
    fit_method="Linear",
):

    '''
    Sort models according to multi metrics, in descending order of goodness.
    '''

    # evaluate model
    result = {}
    for job in os.listdir(model_path):
        if job.startswith("job_"):
            model = torch.load(
                os.path.join(model_path, job, "final.pt"), 
                map_location = device, weights_only=False
            )
            result[job] = evaluate_model(test_ds, model, device=device, fit_method=fit_method)
                         
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
    n_aux=None,
    aux_names=None,
):
    """
    Builds a score matrix with columns:
      [Inter-style Corr, Reconstruction Err, SpearmanCorr(aux1), ..., SpearmanCorr(aux_n)]
    """
    # infer n_aux if not provided
    if n_aux is None:
        if len(result_dict) == 0:
            n_aux = 0
        else:
            first = next(iter(result_dict.values()))
            n_aux = len(first.get("Style-descriptor Corr", {}))

    if aux_names is None:
        aux_names = [f"AUX_{i+1}" for i in range(n_aux)]
    else:
        if len(aux_names) != n_aux:
            raise ValueError(f"aux_names must have length n_aux={n_aux}, got {len(aux_names)}")

    score_names = (
        ["Inter-style Corr", "Reconstruction Err"]
        + [f"Style_{i+1} - {aux_names[i]} Spearman" for i in range(n_aux)]
    )

    scores = []
    jobs = []

    for job, result in result_dict.items():
        jobs.append(job)

        row = [
            result.get("Inter-style Corr", 0.0),
            result.get("Reconstruct Err", [0.0])[0],
        ]

        corr_dict = result.get("Style-descriptor Corr", {})
        for i in range(n_aux):
            ci = corr_dict.get(i, None)
            row.append(0.0 if (ci is None) else float(ci.get("Spearman", 0.0)))

        scores.append(row)

    jobs = np.array(jobs)
    scores = np.array(scores, dtype=float)

    mu_std = np.stack((scores.mean(axis=0), scores.std(axis=0)), axis=1)
    z_scores = (scores - mu_std[:, 0]) / mu_std[:, 1]
    z_scores[:, (mu_std[:, 1] == 0)] = 0.0

    if callable(sort_score):
        final_score = sort_score(z_scores)
    elif isinstance(sort_score, int) and sort_score >= 0:
        final_score = scores[:, sort_score]
    else:
        final_score = np.arange(len(scores), dtype=float)

    rank = np.argsort(final_score)
    if (sort_score is not None) and (not ascending):
        rank = rank[::-1]

    ranked_scores = scores[rank]
    ranked_final_scores = final_score[rank]
    ranked_jobs = jobs[rank]
    ranked_z_scores = z_scores[rank]

    for i, (job, score) in enumerate(zip(ranked_jobs, ranked_final_scores)):
        result_dict[job]["Rank"] = i
        result_dict[job]["Score"] = round(float(score), 4)

    fig = None
    if plot_score:
        if top_n is None or top_n > len(ranked_z_scores):
            top_n = len(ranked_z_scores)

        fig, ax = plt.subplots(figsize=(top_n, scores.shape[1]))
        ax.autoscale(enable=True)
        sns.heatmap(
            ranked_z_scores[:top_n].T,
            vmin=-3, vmax=3,
            cmap="Blues", cbar=True,
            annot=ranked_z_scores[:top_n].T if not true_value else ranked_scores[:top_n].T,
            ax=ax,
            yticklabels=[
                f"{name}\n{ms[0]:.3f}+-{ms[1]:.3f}" for name, ms in zip(score_names, mu_std)
            ],
            xticklabels=[
                f"{ranked_jobs[i]}: {ranked_final_scores[i]:.2f}" for i in range(top_n)
            ],
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left", va="bottom")
        ax.tick_params(labelbottom=False, labeltop=True, axis="both", length=0, labelsize=15)

    return result_dict, ranked_jobs, fig

   
def get_max_inter_style_correlation(styles):
    """
    Maximum absolute Spearman correlation between any pair of style dimensions.
    """
    if styles.shape[1] < 2:
        return 0.0
    corr_list = [
        math.fabs(spearmanr(*styles[:, pair].T).correlation)
        for pair in itertools.combinations(range(styles.shape[1]), 2)
    ]
    return round(max(corr_list), 4)

    
    
def get_descriptor_style_correlation(
    style,
    descriptor,
    ax=None,
    choice=["R2", "Spearman"],
    fit=True
):
    """
    choice can include:
      - "Spearman"
      - "R2" (backward compatible alias for Linear)
      - "Linear"
      - "Quadratic"
      - "Isotonic"
    If ax is given and fit=True, plots the fitted curve from the chosen method.
    """

    # sort by style for plotting + isotonic fitting
    sorted_index = np.argsort(style)
    style = style[sorted_index]
    descriptor = descriptor[sorted_index]

    # mask out NaNs
    mask_nan = ~(np.isnan(descriptor) | np.isnan(style))
    style = style[mask_nan]
    descriptor = descriptor[mask_nan]

    accuracy = {
        "Spearman": None,
        "Linear": {"slope": None, "intercept": None, "R2": None},
        "Quadratic": {"Parameters": [None, None, None], "residue": None, "R2": None},
        "Isotonic": {"R2": None, "residue": None},
    }

    fitted_value = None  # what we will plot if ax is not None

    # Spearman
    if "Spearman" in choice:
        sm = spearmanr(style, descriptor).correlation
        accuracy["Spearman"] = np.round(float(sm), 4).tolist()

    # --- Linear (backward compatible with "R2") ---
    if ("R2" in choice) or ("Linear" in choice):
        result = stats.linregress(style, descriptor)
        fitted_value = result.intercept + style * result.slope

        accuracy["Linear"]["intercept"] = np.round(float(result.intercept), 4).tolist()
        accuracy["Linear"]["slope"] = np.round(float(result.slope), 4).tolist()
        accuracy["Linear"]["R2"] = np.round(float(r2_score(descriptor, fitted_value)), 4).tolist()

    # --- Quadratic ---
    if "Quadratic" in choice:
        p, info = Polynomial.fit(style, descriptor, 2, full=True)
        p_conv = p.convert()
        fitted_value = p_conv(style)

        accuracy["Quadratic"]["Parameters"] = np.round(p_conv.coef, 4).tolist()
        # info[0] is SSE (sum of squared residuals) from numpy fit
        if len(info) > 0 and len(info[0]) > 0:
            accuracy["Quadratic"]["residue"] = np.round(float(info[0][0] / len(style)), 4).tolist()
        accuracy["Quadratic"]["R2"] = np.round(float(r2_score(descriptor, fitted_value)), 4).tolist()

    # --- Isotonic (monotone non-linear) ---
    if "Isotonic" in choice:
        if len(style) < 2 or np.allclose(style, style[0]):
            # degenerate x -> best you can do is constant prediction
            fitted_value = np.full_like(descriptor, float(np.mean(descriptor)), dtype=float)
        else:
            ir = IsotonicRegression(increasing="auto", out_of_bounds="clip")
            fitted_value = ir.fit_transform(style, descriptor)

        accuracy["Isotonic"]["R2"] = np.round(float(r2_score(descriptor, fitted_value)), 4).tolist()
        # store mean squared error as "residue" (similar spirit to quadratic)
        accuracy["Isotonic"]["residue"] = np.round(float(np.mean((descriptor - fitted_value) ** 2)), 4).tolist()

    # plot
    if ax is not None:
        ax.scatter(style, descriptor, s=10.0, c="blue", edgecolors="none", alpha=0.8)
        if fit and (fitted_value is not None):
            ax.plot(style, fitted_value, lw=2, c="black", alpha=0.6)

    return accuracy


def evaluate_model(
    test_ds,
    model,
    reconstruct=True,
    accuracy=True,
    style=True,
    device=torch.device('cpu'),
    fit_method="Linear",
):

    '''
    calculate reconstruction error for a given model, or accuracy.
    
    Returns:
    --------
    '''
    descriptors = test_ds.aux
    result = {
        "Style-descriptor Corr": {},
        "Input": None, 
        "Output": None,
        "Reconstruct Err": (None, None),
        "Inter-style Corr": None  # Inter-style correlation
    }
    
    encoder = model['Encoder']
    decoder = model['Decoder']
    encoder.eval()
    
    # Get styles via encoder
    spec_in = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    styles = encoder(spec_in)
    result["Input"] = spec_in.cpu().numpy()

    if reconstruct:
        spec_out = decoder(styles).clone().detach().cpu().numpy()
        mae_list = []
        for s1, s2 in zip(spec_in.cpu().numpy(), spec_out):
            mae_list.append(mean_absolute_error(s1, s2))
        result["Reconstruct Err"] = [
            round(np.mean(mae_list).tolist(),4),
            round(np.std(mae_list).tolist(),4)
        ]
        result["Output"] = spec_out

    if accuracy:
        styles = styles.clone().detach().cpu().numpy()
        if descriptors is None:
            n_aux = 0
        else:
            n_aux = descriptors.shape[1]

        for i in range(n_aux):
            # style = styles[:, i], descriptor = descriptors[:, i]
            result["Style-descriptor Corr"][i] = get_descriptor_style_correlation(
                styles[:, i],
                descriptors[:, i],
                ax=None,
                choice=["Spearman", fit_method]
            )

    

    if style:
        result["Inter-style Corr"] = get_max_inter_style_correlation(styles)

    return result


def qqplot_normal(x, ax=None, grid=True):
    """
    Examine the "normality" of a distribution using qqplot.
    Return the Shapiro statistic that represent the similarity of `x` to normality.
    """
    data_length = len(x)
    
    # standardize input data, and calculate the z-score
    x_std = (x - x.mean())/x.std()
    z_score = sorted(x_std)
    
    # sample from standard normal distribution and calculate quantiles
    normal = np.random.randn(data_length)
    q_normal = np.quantile(normal, np.linspace(0,1,data_length))

    # Calculate Shapiro statistic for z_score
    shapiro_statistic = shapiro(z_score).statistic
    # make the q-q plot if ax is given
    if ax is not None:
        ax.plot(q_normal, z_score, ls='',marker='.', color='k')
        ax.plot([q_normal.min(),q_normal.max()],[q_normal.min(),q_normal.max()],
                 color='k',alpha=0.5)
        ax.grid(grid)
    return shapiro_statistic
