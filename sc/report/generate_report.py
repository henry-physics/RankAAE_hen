import torch
import numpy as np
import pickle
from collections import OrderedDict
from matplotlib import pyplot as plt
import os
import argparse
import json
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import sc.report.analysis as analysis
import sc.report.analysis_new as analysis_new
from sc.utils.parameter import Parameters
from sc.clustering.dataloader import AuxSpectraDataset


def sorting_algorithm(x):
    """
    x columns are:
      0: Inter-style Corr (lower is better)
      1: Reconstruction Err (lower is better)
      2..: Spearman correlations (higher is better)
    Input x is typically z-scores from analysis.sort_all_models().
    """
    if x.shape[1] < 2:
        return np.zeros((x.shape[0],), dtype=float)

    w = np.ones(x.shape[1], dtype=float)
    w[0] = -1.0  # penalize inter-style corr
    w[1] = -1.0  # penalize recon err
    return (x * w).sum(axis=1)

def plot_descriptor_style_scatter_matrix(
    styles,              # (N, n_styles)
    descriptors,         # (N, n_aux)
    aux_names=None,
    fit_method="Linear",     # curve to DRAW: "Linear", "Quadratic", "Isotonic"
    fit=True,
    max_points=None,
    use_square=True,         # if True -> N×N with N=min(n_styles,n_aux)
    title="Descriptor vs Style scatter matrix",
):
    if descriptors is None:
        return None

    styles = np.asarray(styles)
    descriptors = np.asarray(descriptors)

    n_styles = styles.shape[1]
    n_aux = descriptors.shape[1]

    if aux_names is None:
        aux_names = [f"AUX_{i+1}" for i in range(n_aux)]
    if len(aux_names) != n_aux:
        raise ValueError(f"aux_names must have length n_aux={n_aux}, got {len(aux_names)}")

    # optional subsample
    Npts = styles.shape[0]
    if (max_points is not None) and (Npts > max_points):
        idx = np.random.choice(Npts, size=max_points, replace=False)
        styles = styles[idx]
        descriptors = descriptors[idx]

    # choose which block to plot
    if use_square:
        n = min(n_styles, n_aux)
        styles_plot = styles[:, :n]
        desc_plot = descriptors[:, :n]
        aux_names_plot = aux_names[:n]
        n_rows, n_cols = n, n
    else:
        styles_plot = styles
        desc_plot = descriptors
        aux_names_plot = aux_names
        n_rows, n_cols = n_aux, n_styles

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
        dpi=120
    )
    axes = np.atleast_2d(axes)
    fig.suptitle(title)

    for i_aux in range(n_rows):
        for j_style in range(n_cols):
            ax = axes[i_aux, j_style]
            x = styles_plot[:, j_style]
            y = desc_plot[:, i_aux]

            # Always compute Spearman + Linear (for R^2), and optionally another fit to draw
            choice = ["Spearman", fit_method]

            draw_fit = fit and (i_aux == j_style)

            acc = analysis.get_descriptor_style_correlation(
                x, y,
                ax=ax,
                choice=choice,
                fit=draw_fit
            )
            
            sp = acc.get("Spearman", None)
            r2 = acc.get(fit_method, {}).get("R2", None)

            sp_str = "NA" if sp is None else f"{sp:.2f}"
            r2_str = "NA" if r2 is None else f"{r2:.2f}"

            ax.set_title(
                f"S{j_style+1} vs {aux_names_plot[i_aux]}\n"
                f"Spearman={sp_str}, {fit_method} R2={r2_str}",
                fontsize=9
            )
            

            if i_aux == n_rows - 1:
                ax.set_xlabel(f"style_{j_style+1}")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if j_style == 0:
                ax.set_ylabel(aux_names_plot[i_aux])
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    return fig


def plot_report(test_ds, model, config=None, title="report", device=torch.device("cpu")):
    n_aux = int(getattr(config, "n_aux", 0))
    aux_names = getattr(config, "aux_names", None)
    if aux_names is None:
        aux_names = [f"AUX_{i+1}" for i in range(n_aux)]
    else:
        if len(aux_names) != n_aux:
            raise ValueError(f"aux_names must have length n_aux={n_aux}, got {len(aux_names)}")

    plot_residual = bool(getattr(config, "plot_residual", False))
    n_sampling = int(getattr(config, "n_sampling", 1000))

    encoder = model["Encoder"]
    decoder = model["Decoder"]

    # evaluate for the title only
    result = analysis.evaluate_model(test_ds, model, device=device)
    style_correlation = result.get("Inter-style Corr", 0.0)

    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    test_grid = test_ds.grid
    test_styles = encoder(test_spec).detach().cpu().numpy()
    n_styles = test_styles.shape[1]


     

    descriptors = test_ds.aux
    n_scatter = 0 if descriptors is None else descriptors.shape[1]
    n_scatter = min(n_scatter, n_styles)

    # Layout: style-variation plots + one scatter per aux (aux_i vs style_i)
    ncols = 2
    nrows_style = int(np.ceil(n_styles / ncols))
    nrows_scatter = int(np.ceil(n_scatter / ncols)) if n_scatter > 0 else 0
    total_rows = nrows_style + nrows_scatter

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(12, 4 * total_rows),
        constrained_layout=True,
        dpi=100
    )
    axes = np.atleast_2d(axes)
    flat_axes = axes.reshape(-1)

    fig.suptitle(
        f"{title}\nMax |Spearman| between styles: {style_correlation:.4f}"
    )

    # ---- style variation ----
    for istyle in range(n_styles):
        ax = flat_axes[istyle]
        analysis.plot_spectra_variation(
            decoder,
            istyle,
            true_range=True,
            styles=test_styles,
            amplitude=2,
            n_spec=50,
            n_sampling=n_sampling,
            device=device,
            energy_grid=test_grid,
            plot_residual=plot_residual,
            ax=ax,
        )

    # hide unused axes in style section
    for ax in flat_axes[n_styles : nrows_style * ncols]:
        ax.axis("off")

    # ---- scatter plots: aux_i vs style_i ----
    fit_method = getattr(config, "fit_method", "Linear")
    for i in range(n_scatter):
        ax = flat_axes[nrows_style * ncols + i]
        x = test_styles[:, i]
        y = descriptors[:, i]
        acc = analysis.get_descriptor_style_correlation(
            x, y, ax=ax, choice=["Spearman", fit_method], fit=True
        )
        ax.set_xlabel(f"style_{i+1}")
        ax.set_ylabel(aux_names[i])
        r2 = acc.get(fit_method, {}).get("R2", None)
        r2_str = "NA" if (r2 is None) else f"{r2:.2f}"
        ax.set_title(
            f"{aux_names[i]} vs style_{i+1}: Spearman={acc['Spearman']:.2f}, {fit_method} R2={r2_str}"
        )
                
                

    # hide remaining axes
    for ax in flat_axes[nrows_style * ncols + n_scatter :]:
        ax.axis("off")

    return fig

    

def save_evaluation_result(save_dir, file_name, model_results, save_spectra=False, top_n=5):
    """
    Input is a dictionary of result dictionaries of evaluate_model.
    And file name to save the resul.
    Information is saved to a txt file.
    """
    save_dict = OrderedDict()
    if top_n > len(model_results):
        top_n = len(model_results)
    sorted_top_n_jobs = list(range(top_n))
    for job, result in model_results.items():
        if result['Rank'] in sorted_top_n_jobs:
            sorted_top_n_jobs[result['Rank']] = job
    for job in sorted_top_n_jobs:
        result = model_results[job]
        save_dict[job] = {
            k: v for k, v in result.items() if k not in ["Input", "Output"]
        }
        if (result['Rank'] == 0) and save_spectra:
            spec_in = result["Input"]
            spec_out = result["Output"]
    with open(os.path.join(save_dir, file_name+'.json'), 'wt') as f:
        f.write(json.dumps(save_dict))
    np.savetxt(os.path.join(save_dir, file_name+'.out'),spec_out)
    np.savetxt(os.path.join(save_dir, file_name+'.in'),spec_in)


def save_model_evaluations(save_dir, file_name, result):
    with open(os.path.join(save_dir, file_name+"_model_evaluation.pkl"), "wb") as f:
        pickle.dump(result, f)


def save_model_selection_plot(save_dir, file_name, fig):
    fig.savefig(
        os.path.join(save_dir, file_name + "_model_selection.png"),
        bbox_inches = 'tight'
    )


def main():
    #### Parse arguments ####
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--work_dir', type = str, default = '.',
                        help = "The folder where the model and data are.")  
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config for training parameter in YAML format')
    
    args = parser.parse_args()
    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    config = Parameters.from_yaml(os.path.join(work_dir, args.config))
    

    jobs_dir = os.path.join(work_dir, "training")
    file_name = config.data_file
    
    device = torch.device("cpu") # device is cpu by default
    if config.gpu: 
        try:
            device = torch.device("cuda:0")
        except:
            device = torch.device("cpu")

    #### Create test data set from file ####
    if file_name == None:  # if datafile name nor provided, search for it.
        data_file_list = [f for f in os.listdir(work_dir) if f.endswith('.csv')]
        assert len(data_file_list) == 1, "Which data file are you going to use?"
        file_name = data_file_list[0]
    test_ds = AuxSpectraDataset(
        os.path.join(work_dir, file_name),
        split_portion="val",
        n_aux=config.n_aux,
        shuffle=getattr(config, "shuffle_data", True),
        random_seed=getattr(config, "random_seed", 0),
    )
     
    fit_method = getattr(config, "fit_method", "Linear")

    plot_job = getattr(config, "plot_job", None)
    aux_names = getattr(config, "aux_names", None)

    if plot_job is not None:
        sorted_jobs = [plot_job]
    else:
        model_results = analysis.evaluate_all_models(jobs_dir, test_ds, device=device,fit_method=fit_method)
        model_results, sorted_jobs, fig_model_selection = analysis.sort_all_models(
            model_results,
            plot_score=True,
            top_n=config.top_n,
            sort_score=sorting_algorithm,
            ascending=False,
            n_aux=config.n_aux,
            aux_names=aux_names,
        )
        

        save_model_evaluations(work_dir, config.output_name, model_results)

        if fig_model_selection is not None:
            save_model_selection_plot(work_dir, config.output_name, fig_model_selection)

        save_evaluation_result(work_dir, config.output_name, model_results, save_spectra=True, top_n=config.top_n)

    # ---- common path from here on ----
    job0 = sorted_jobs[0]
    output_path_best_model = os.path.join(work_dir, f"{config.output_name}_{job0}.png")

    top_model = torch.load(
        os.path.join(jobs_dir, job0, "final.pt"),
        map_location=device,
        weights_only=False
    )

    fig_top_model = plot_report(
        test_ds,
        top_model,
        config=config,
        title='-'.join([config.output_name, job0]),
        device=device
    )

    fig_top_model.savefig(output_path_best_model, bbox_inches="tight")
        
    encoder = top_model["Encoder"]
    encoder.eval()
    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    test_styles = encoder(test_spec).detach().cpu().numpy()

    fig_matrix = plot_descriptor_style_scatter_matrix(
        test_styles,
        test_ds.aux,
        aux_names=getattr(config, "aux_names", None),
        fit_method=getattr(config, "fit_method", "Linear"),
        fit=True,              # set True if you want fit curves everywhere
        max_points=None,        # e.g. 2000 if large
        title='-'.join([config.output_name, job0, "scatter_matrix"])
    )

    if fig_matrix is not None:
        matrix_path = os.path.join(work_dir, f"{config.output_name}_{job0}_scatter_matrix.png")
        fig_matrix.savefig(matrix_path, bbox_inches="tight")

    recon_evaluator = analysis_new.Reconstruct(name=config.output_name, device=device)
    recon_evaluator.evaluate(test_ds, top_model, path_to_save=work_dir)
    
    
    
    plotter = analysis_new.LossCurvePlotter()
    fig = plotter.plot_loss_curve(os.path.join(jobs_dir, sorted_jobs[0], "losses.csv"))
    fig.savefig("loss_curves.png", bbox_inches="tight")
    print("Success: training report saved!")

if __name__ == "__main__":
    main()
