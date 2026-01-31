#!/usr/bin/env python

import argparse

import torch
from sc.clustering.trainer import Trainer
from sc.utils.parameter import Parameters
from sc.utils.logger import create_logger
import os
import yaml
import socket
import ipyparallel as ipp
import logging
import signal
import time
import numpy as np


engine_id = -1

def timeout_handler(signum, frame):
    raise Exception("Training Overtime!")


def get_parallel_map_func(work_dir=".", logger=logging.getLogger("Parallel")):
    
    c = ipp.Client(
        connection_info=f"{work_dir}/ipypar/security/ipcontroller-client.json"
    )

    with c[:].sync_imports():
        import torch
        from sc.clustering.trainer import Trainer
        from sc.utils.parameter import Parameters
        from sc.utils.logger import create_logger
        import os
        import socket
        import logging
        import signal
        import time
    logger.info(f"Engine IDs: {c.ids}")
    c[:].push(dict(run_training=run_training, timeout_handler=timeout_handler),
              block=True)

    return c.load_balanced_view().map_sync, len(c.ids)


def run_training(
    job_number, 
    work_dir, 
    train_config,  
    verbose, 
    data_file, 
    timeout_hours=0,
    logger = logging.getLogger("training"),
    callback=None,

):

    work_dir = f'{work_dir}/training/job_{job_number+1}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # Set up a logger to record general training information
    logger = create_logger(f"subtraining_{job_number+1}", os.path.join(work_dir, "messages.txt"))
    
    # Set up a logger to record losses against epochs during training 
    loss_logger = create_logger(f"losses_{job_number+1}", os.path.join(work_dir, "losses.csv"), simple_fmt=True)

    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
    
    ngpus_per_node = torch.cuda.device_count()
    if "SLURM_LOCALID" in os.environ:
        local_id = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        local_id = 0
    igpu = local_id % ngpus_per_node if torch.cuda.is_available() else -1
    
    start = time.time()
    logger.info(f"Training started for trial {job_number+1}.")

    trainer = Trainer.from_data(
        data_file,
        igpu = igpu,
        verbose = verbose,
        work_dir = work_dir,
        config_parameters = train_config,
        logger = logger,
        loss_logger = loss_logger,
    )
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_hours * 3600))

    metrics = trainer.train(callback=callback)
    logger.info(metrics)

    signal.alarm(0)
    
    time_used = time.time() - start
    logger.info(f"Training finished. Time used: {time_used:.2f}s.\n\n")
    
    return metrics, time_used


# optuna 

def suggest_from_space(trial, space: dict):
    """
    space example:
      lr_base: {type: loguniform, low: 1e-4, high: 1e-2}
      dropout_rate: {type: uniform, low: 0.0, high: 0.2}
      n_layers: {type: int, low: 3, high: 7}
      optimizer_name: {type: categorical, choices: [AdamW, Adam]}
    """
    params = {}
    for name, spec in space.items():
        t = spec["type"].lower()
        if t == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif t == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown optuna type '{spec['type']}' for param '{name}'")
    return params


def make_trial_parameters(base_params: Parameters, trial_params: dict) -> Parameters:
    base_dict = dict(base_params.to_dict())  # copy
    base_dict.update(trial_params)
    return Parameters(base_dict)


def make_optuna_callback(trial, trainer_obj):
    # trainer_obj is only used for metric_weights if you want consistency; otherwise omit it
    def cb(epoch, metrics):
        # same combined metric definition as Trainer uses
        w = np.array(trainer_obj.metric_weights, dtype=float)
        m = np.array(metrics, dtype=float)
        combined_metric = - (w * m).sum()

        trial.report(combined_metric, step=epoch)
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}")
    return cb


def run_optuna_trial(
    trial_id,
    trial_number,
    work_dir,
    base_train_config,
    verbose,
    data_file,
    timeout_hours,
    trial_params: dict,
    storage: str,
    study_name: str,
    direction: str,
):
    import optuna

    # load the same study (must include same pruner config if you set one)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=40, interval_steps=10),
    )

    trial = optuna.trial.Trial(study, trial_id)

    trial_config = make_trial_parameters(base_train_config, trial_params)
    

    # (mostly your run_training code, but we need trainer object to build callback)
    subdir = f'{work_dir}/training/job_{trial_number+1}'
    os.makedirs(subdir, exist_ok=True)

    logger = create_logger(f"subtraining_{trial_number+1}", os.path.join(subdir, "messages.txt"))
    loss_logger = create_logger(f"losses_{trial_number+1}", os.path.join(subdir, "losses.csv"), simple_fmt=True)

    ngpus_per_node = torch.cuda.device_count()
    local_id = int(os.environ.get("SLURM_LOCALID", 0)) if "SLURM_LOCALID" in os.environ else 0
    igpu = local_id % ngpus_per_node if torch.cuda.is_available() else -1

    trainer = Trainer.from_data(
        data_file,
        igpu=igpu,
        verbose=verbose,
        work_dir=subdir,
        config_parameters=trial_config,
        logger=logger,
        loss_logger=loss_logger,
    )

    cb = make_optuna_callback(trial, trainer)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_hours * 3600))
    start = time.time()

    # IMPORTANT: this can raise optuna.TrialPruned
    result_dict = trainer.train(callback=cb)

    signal.alarm(0)
    time_used = time.time() - start

    objective = result_dict["best_combined_metric"]
    return objective, result_dict, time_used, trial_params




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config for training parameter in YAML format')
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory to write the output files")
    args = parser.parse_args()

    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    train_config = Parameters.from_yaml(os.path.join(work_dir, args.config))
    assert os.path.exists(work_dir)

    ### optuna 
    optuna_cfg = train_config.get("optuna", None)
    optuna_enabled = bool(optuna_cfg and optuna_cfg.get("enabled", False))

    
    ###

    verbose = train_config.get("verbose", False)
    trials = train_config.get("trials", 1)
    data_file = os.path.join(work_dir, train_config.get("data_file", None))
    timeout = train_config.get("timeout", 10)

    # Start Logger
    logger = create_logger("Main training:", f'{work_dir}/main_process_message.txt', append=True)
    logger.info("START")

    ### replace if trails > 1

    if optuna_enabled:
        import optuna

        space = optuna_cfg["search_space"]
        n_trials = int(optuna_cfg.get("n_trials", 20))
        n_parallel = int(optuna_cfg.get("n_parallel", 1))

        storage = optuna_cfg.get("storage", f"sqlite:///{work_dir}/optuna.db")
        study_name = optuna_cfg.get("study_name", "sc_study")
        direction = optuna_cfg.get("direction", "minimize")

        ### pruner 

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,     # don’t prune until you have some history
            n_warmup_steps=40,      # don’t prune in the first N epochs
            interval_steps=10        # check every N epochs
        )
        

        ###

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True,
            pruner=pruner,
        )

        # start ipyparallel view
        c = ipp.Client(connection_info=f"{work_dir}/ipypar/security/ipcontroller-client.json")
        lbv = c.load_balanced_view()
        logger.info(f"Optuna enabled. Engines: {c.ids}")

        # push needed functions to engines
        c[:].push(dict(
            run_training=run_training,
            run_optuna_trial=run_optuna_trial,
            make_trial_parameters=make_trial_parameters,
            suggest_from_space=suggest_from_space,
            timeout_handler=timeout_handler,
        ), block=True)

        running = {}  # async_result -> (trial, trial_params)

        completed = 0
        while completed < n_trials:
            # launch up to n_parallel concurrent trials
            while len(running) < n_parallel and completed + len(running) < n_trials:
                trial = study.ask()
                trial_params = suggest_from_space(trial, space)

                ar = lbv.apply_async(
                    run_optuna_trial,
                    trial._trial_id,
                    trial.number,
                    work_dir,
                    train_config,
                    verbose,
                    data_file,
                    timeout,
                    trial_params,
                    storage, 
                    study_name, 
                    direction,
                )
                running[ar] = (trial, trial_params)

            # wait for at least one to finish
            done = [ar for ar in list(running.keys()) if ar.ready()]
            if not done:
                time.sleep(1.0)
                continue

            for ar in done:
                trial, trial_params = running.pop(ar)
                try:

                    objective, result_dict, time_used, _ = ar.get()
                    study.tell(trial, objective)
                    logger.info(...)
                except optuna.TrialPruned as e:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    logger.info(f"[Optuna] trial={trial.number} pruned: {e}")
                except Exception as e:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    logger.exception(...)


                completed += 1

        import optuna

        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete_trials:
            logger.error("[Optuna] No COMPLETE trials. Check earlier errors/logs.")
            return


        logger.info(f"[Optuna] Best trial: {study.best_trial.number} value={study.best_value} params={study.best_params}")

        # write best params to file (so report step can use it)
        best_yaml = dict(train_config.to_dict())
        best_yaml.update(study.best_params)
        best_yaml["optuna"] = dict(optuna_cfg)
        best_yaml["optuna"]["enabled"] = False  # freeze for reruns/reports

        best_config_path = os.path.join(work_dir, "best_config.yaml")
        with open(best_config_path, "w") as f:
            yaml.safe_dump(best_yaml, f, sort_keys=False)

        with open(os.path.join(work_dir, "best_trial.txt"), "w") as f:
            f.write(str(study.best_trial.number))

        return


    ###

    if not optuna_enabled:

        if trials > 1:
            par_map, nprocesses = get_parallel_map_func(work_dir, logger=logger)
        else:
            par_map, nprocesses = map, 1
        logger.info("Running with {} process(es).".format(nprocesses))
        
        start = time.time()
        result = par_map(
            run_training,
            list(range(trials)),
            [work_dir] * trials,
            [train_config] * trials,
            [verbose] * trials,
            [data_file] * trials,
            [timeout] * trials,
            [logger] * trials
        )

        time_trials = np.array([r[1] for r in list(result)])
        logger.info(
            f"Time used for each trial: {time_trials.mean():.2f} +/- {time_trials.std():.2f}s.\n" + 
            ' '.join([f"{t:.2f}s" for t in time_trials])
        )
        
        end = time.time()
        logger.info(
            f"Total time used: {end-start:.2f}s for {trials} trails " +
            f"({(end-start)/trials:.2f} each on average)."
        )
        logger.info("END\n\n")



if __name__ == '__main__':
    main()
