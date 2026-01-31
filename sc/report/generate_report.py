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
import re 


def _safe_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'[^a-zA-Z0-9._-]+', '_', s)
    return s[:120]

def save_diagonal_scatter_pngs(test_ds, model, config, out_dir, device=torch.device("cpu")):
    os.makedirs(out_dir, exist_ok=True)

    encoder = model['Encoder']

    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    test_styles = encoder(test_spec).detach().cpu().numpy()
    descriptors = np.asarray(test_ds.aux)

    n_pair = min(test_styles.shape[1], descriptors.shape[1])
    test_styles = test_styles[:, :n_pair]
    descriptors = descriptors[:, :n_pair]

    descriptor_names = getattr(config, "descriptor_names",
                               [f"Descriptor_{i}" for i in range(descriptors.shape[1])])

    discrete_idx = getattr(config, "discrete_idx", None)
    if discrete_idx is not None:
        discrete_idx = int(discrete_idx)
        if discrete_idx < 0 or discrete_idx >= n_pair:
            discrete_idx = None

    # only continuous dims on diagonal (skip discrete, if any)
    if discrete_idx is None:
        cont_idx = list(range(n_pair))
    else:
        cont_idx = [i for i in range(n_pair) if i != discrete_idx]

    styles_cont = test_styles[:, cont_idx]
    desc_cont   = descriptors[:, cont_idx]
    names_cont  = [descriptor_names[i] for i in cont_idx]

    for k in range(len(cont_idx)):
        style_i = cont_idx[k]
        name = names_cont[k]

        fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=160)
        acc = analysis.get_descriptor_style_correlation(
            styles_cont[:, k],
            desc_cont[:, k],
            ax=ax,
            choice=["R2", "Spearman"],
            fit=True,
        )

        r2 = acc["Linear"]["R2"]
        sp = acc["Spearman"]
        ax.set_title(f"{name} vs style_{style_i + 1}\nR2={r2:.3f}, Spearman={sp:.3f}", fontsize=10)

        out_path = os.path.join(out_dir, f"diag_{k+1:02d}_style{style_i+1:02d}_{_safe_filename(name)}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)



def sorting_algorithm(x, inter_weight=-1.0, recon_power=0.0, desc_weights=None, eps=1e-12):
    """
    x columns:
      0: Inter-style Corr        (want small)
      1: Reconstruction Err      (want small)
      2..: Style-Descriptor Corr (want large)
    """

    x = np.asarray(x)
    n = x.shape[1]
    assert n >= 2, "Need at least [inter_corr, recon_err] columns"

    # descriptor weights (one per descriptor column, columns 2..end)
    n_desc = max(0, n - 2)
    if desc_weights is None:
        desc_weights = np.ones(n_desc)          # default: treat new descriptors equally
    else:
        desc_weights = np.asarray(desc_weights, dtype=float)
        # pad/truncate to match number of descriptor columns
        if len(desc_weights) < n_desc:
            desc_weights = np.pad(desc_weights, (0, n_desc - len(desc_weights)), constant_values=1.0)
        else:
            desc_weights = desc_weights[:n_desc]

    inter_term = inter_weight * x[:, 0]

    if n_desc > 0:
        desc_term = (x[:, 2:] * desc_weights).sum(axis=1)
    else:
        desc_term = 0.0

    numerator = inter_term + desc_term

    # If you ever set all weights so numerator becomes 0 for everything,
    # keep an offset so score isn't identically 0.
    offset = 1.0 if np.allclose(numerator, 0.0) else 0.0

    recon = np.clip(x[:, 1], eps, None)
    denom = recon ** recon_power   # recon_power=0 => denom=1 (ignore recon)

    return (offset + numerator) / denom


### additional function to save individual plots 

def save_style_variation_png(test_ds, model, config, out_path, device):
    encoder = model["Encoder"]
    decoder = model["Decoder"]

    spec = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    styles = encoder(spec).detach().cpu().numpy()
    grid = test_ds.grid

    n_styles = styles.shape[1]
    n_spec = 50

    # pick your palette (see next section)
    colors = plt.cm.viridis(np.linspace(0, 1, n_spec))  # RGBA array is fine

    fig, axs = plt.subplots(
        n_styles, 1,
        figsize=(10, 3*n_styles),
        constrained_layout=True,
        dpi=150,
        sharex=True
    )
    if n_styles == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        analysis.plot_spectra_variation(
            decoder, i,
            true_range=True,
            styles=styles,
            n_spec=n_spec,
            n_sampling=config.n_sampling,
            device=device,
            energy_grid=grid,
            colors=colors,          # <---- custom colors here
            plot_residual=getattr(config, "plot_residual", False),
            ax=ax
        )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    
    for i in range(n_styles):

        true_range = True 
        amplitude = [0,0]

        if i == 0: 
            true_range = False 
            amplitude = [0,2]
        if i ==3: 
            true_range=False 
            amplitude=[-2,1]

        if i ==9: 
            true_range=False 
            amplitude = [-3,2]



        fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
        
        analysis.plot_spectra_variation(
            decoder, i,
            true_range=true_range,
            styles=styles,
            n_spec=n_spec,
            n_sampling=config.n_sampling,
            device=device,
            energy_grid=grid,
            amplitude=amplitude, 
            colors=colors,
            plot_residual=getattr(config, "plot_residual", False),
            ax=ax
        )
        
        #plt.show()  # Show each plot individually
        plt.close(fig)

def plot_report(test_ds, model, config=None, title='report', device=torch.device("cpu")):
    plot_residual = getattr(config, "plot_residual", False)

    encoder = model['Encoder']
    decoder = model['Decoder']

    # --- data ---
    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    test_grid = test_ds.grid
    test_styles = encoder(test_spec).detach().cpu().numpy()
    descriptors = np.asarray(test_ds.aux)

    # truncate to matching dims (prevents indexing surprises)
    n_pair = min(test_styles.shape[1], descriptors.shape[1])
    test_styles = test_styles[:, :n_pair]
    descriptors = descriptors[:, :n_pair]

    # names (YOU provide via config.descriptor_names)
    descriptor_names = getattr(config, "descriptor_names",
                              [f"Descriptor_{i}" for i in range(descriptors.shape[1])])

    # discrete_idx == None (from YAML null) means: no discrete descriptor
    discrete_idx = getattr(config, "discrete_idx", None)
    if discrete_idx is not None:
        discrete_idx = int(discrete_idx)
        if discrete_idx < 0 or discrete_idx >= n_pair:
            discrete_idx = None

    has_discrete = (discrete_idx is not None)

    # continuous indices
    if discrete_idx is None:
        cont_idx = list(range(n_pair))          # all continuous
    else:
        cont_idx = [i for i in range(n_pair) if i != discrete_idx]
        


    styles_cont = test_styles[:, cont_idx]
    desc_cont = descriptors[:, cont_idx]
    names_cont = [descriptor_names[i] for i in cont_idx]
    n_cont = len(cont_idx)

    
    discrete_max_classes = getattr(config, "discrete_max_classes", 10)
    if discrete_idx is None:
        discrete_max_classes = 0   # force: no discrete handling in evaluate_model()

    result = analysis.evaluate_model(
        test_ds, model, device=device,
        discrete_descriptor_max_classes=discrete_max_classes
    )
    


    style_correlation = result["Inter-style Corr"]

    # --- figure layout (minimal change but supports 5x5) ---
    # make enough room: top spectra (2x3) + cont matrix (n_cont x n_cont) + discrete block (3 plots) + qq row
    right_cols = 3 if has_discrete else 0 
    ncols = n_cont + right_cols  # left: n_cont cols, right: 3 cols for discrete plots/qq
    #nrows = 4 + n_cont + 1  # 4 rows for spectra, n_cont rows for matrix, 1 row for qq
    nrows = n_cont + 1  # 4 rows for spectra, n_cont rows for matrix, 1 row for qq
    fig = plt.figure(figsize=(4.2*ncols, 3.6*nrows), constrained_layout=True, dpi=160)
    gs = fig.add_gridspec(nrows, ncols,wspace=0.35,hspace=0.55)

    fig.suptitle(f"{title}\nLeast correlation: {style_correlation:.4f}")

    # --- 6 style variation plots (same idea as before) ---
    #spec_gs = gs[0:4, 0:ncols].subgridspec(2, 3)
    #axs_spec = [fig.add_subplot(spec_gs[r, c]) for r in range(2) for c in range(3)]

    #n_styles = test_styles.shape[1]
    #axs_spec = axs_spec[:min(6, n_styles)]  # keep behavior similar to old code

    #spectra_reconstructed = []
    #for istyle, ax in enumerate(axs_spec):
    #    _, spec_reconstructed = analysis.plot_spectra_variation(
    #        decoder, istyle,
    #        true_range=True,
    #        styles=test_styles,
    #        amplitude=2,
    #        n_spec=50,
    #        n_sampling=config.n_sampling,
    #        device=device,
    #        energy_grid=test_grid,
    #        plot_residual=plot_residual,
    #        ax=ax
    #    )
    #    spectra_reconstructed.append(spec_reconstructed)
#
#    if plot_residual and len(spectra_reconstructed) >= 2:
#        residuals = [s[-1] - s[0] for s in spectra_reconstructed]
#        cos_sim_matrix = cosine_similarity(residuals, residuals)
#        for istyle, ax in enumerate(axs_spec):
#            cos_sim_list = cos_sim_matrix[istyle]
#            max_cos_sim = -1
#            max_jstyle = None
#            for jstyle, cos_sim in enumerate(cos_sim_list):
#                if jstyle == istyle:
#                    continue
#                if cos_sim >= max_cos_sim:
#                    max_cos_sim = cos_sim
#                    max_jstyle = jstyle
#            ax.text(0.95, 0.95, f"max_cos_sim: {max_cos_sim:.2f}\nwith style{max_jstyle+1}",
#                    va="top", ha="right", transform=ax.transAxes, fontsize=12)

    # --- continuous descriptor vs continuous style: NxN grid ---
    mat_gs = gs[0:n_cont, 0:n_cont].subgridspec(n_cont, n_cont,wspace=0.4, hspace=0.6)

    for row in range(n_cont):
        for col in range(n_cont):
            ax = fig.add_subplot(mat_gs[row, col])

            plot_fit = (row == col)
            result_choice = ["R2", "Spearman"]  # keep it simple/robust

            accuracy = analysis.get_descriptor_style_correlation(
                styles_cont[:, col],
                desc_cont[:, row],
                ax=ax,
                choice=result_choice,
                fit=plot_fit,
            )

            r2 = accuracy["Linear"]["R2"]
            sp = accuracy["Spearman"]
            ax.set_title(f"{names_cont[row]}\n {r2}/{sp}", fontsize=7.5, pad=2)
    
    n_qq = n_cont + (1 if has_discrete else 0)
    qq_gs = gs[nrows-1, 0:n_qq].subgridspec(1, n_qq, wspace=0.4)

    for j, idx in enumerate(cont_idx):
        ax = fig.add_subplot(qq_gs[0, j])
        sh = analysis.qqplot_normal(test_styles[:, idx], ax)
        ax.set_title(f"style_{idx}: {sh:.2f}", fontsize=9)

    if has_discrete:
        ax_disc_qq = fig.add_subplot(qq_gs[0, n_cont])
        sh = analysis.qqplot_normal(test_styles[:, discrete_idx], ax_disc_qq)
        ax_disc_qq.set_title(f"style_{discrete_idx} (discrete): {sh:.2f}", fontsize=9)
        



    if has_discrete:
        disc_gs = gs[0:3, n_cont:ncols].subgridspec(3, 1, hspace=0.35)
        ax5 = fig.add_subplot(disc_gs[0, 0])
        ax6 = fig.add_subplot(disc_gs[1, 0])
        ax7 = fig.add_subplot(disc_gs[2, 0])

        _ = analysis.get_confusion_matrix(
            descriptors[:, discrete_idx].astype(int),
            test_styles[:, discrete_idx],
            ax=[ax5, ax6, ax7]
        )
    

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
    test_ds = AuxSpectraDataset(os.path.join(work_dir, file_name), split_portion = "val", n_aux = config.n_aux)
    
    try:
        sorted_jobs = [config.plot_job]
        output_path_best_model = os.path.join(work_dir, f"{config.output_name}_{sorted_jobs[0]}.png")
    except:
        #### Choose the 20 top model based on evaluation criteria ####
        model_results = analysis.evaluate_all_models(jobs_dir, test_ds, device=device) # models are not sorted
        model_results, sorted_jobs, fig_model_selection = analysis.sort_all_models( 
            model_results, 
            plot_score = True, 
            top_n = config.top_n, 
            sort_score = sorting_algorithm,
            ascending = False, # best model has the highest score
        ) # models are sorted
        save_model_evaluations(work_dir, config.output_name, model_results)
        
        # genearte model selection scores plot
        if fig_model_selection is not None:
            save_model_selection_plot(work_dir, config.output_name, fig_model_selection)

        # save top 5 result 
        save_evaluation_result(work_dir, config.output_name, model_results, save_spectra=True, top_n=config.top_n)
        output_path_best_model = os.path.join(work_dir, f"{config.output_name}_best_model.png")
    finally:
        # generate report for top model
        job_dir = os.path.join(jobs_dir, sorted_jobs[0])
        #ckpt = "best.pt" if os.path.exists(os.path.join(job_dir, "best.pt")) else "final.pt"
        ckpt = "best.pt" 
        top_model = torch.load(os.path.join(job_dir, ckpt), map_location=device, weights_only=False)

        diag_out_dir =  os.path.join(work_dir, "plots")

        diag_only = True

        if diag_only:
            save_diagonal_scatter_pngs(test_ds, top_model, config, diag_out_dir, device=device)
            fig_top_model = None
        else:
            fig_top_model = plot_report(
                test_ds,
                top_model,
                config=config,
                title='-'.join([config.output_name, sorted_jobs[0]]),
                device=device
            )
        if fig_top_model is not None:
            fig_top_model.savefig(output_path_best_model, bbox_inches="tight")

    

    out_path_styles = os.path.join(work_dir, f"{config.output_name}_best_model_style_variations.png")
    save_style_variation_png(test_ds, top_model, config, out_path_styles, device)  

    if fig_top_model is not None: 
        fig_top_model.savefig(output_path_best_model, bbox_inches="tight")
    recon_evaluator = analysis_new.Reconstruct(name=config.output_name, device=device)
    recon_evaluator.evaluate(test_ds, top_model, path_to_save=work_dir)
    
    
    
    plotter = analysis_new.LossCurvePlotter()
    fig = plotter.plot_loss_curve(os.path.join(jobs_dir, sorted_jobs[0], "losses.csv"))
    fig.savefig("loss_curves.png", bbox_inches="tight")
    print("Success: training report saved!")

if __name__ == "__main__":
    main()
