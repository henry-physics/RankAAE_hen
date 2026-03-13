import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import optuna
from optuna.importance import get_param_importances

from sc.clustering.trainer import Trainer
from sc.utils.logger import create_logger
from sc.utils.parameter import Parameters


def timeout_handler(signum, frame):
    raise Exception("Training Overtime!")


def run_training(
    job_number,
    work_dir,
    train_config,
    verbose,
    data_file,
    timeout_hours=0,
    logger=logging.getLogger("training"),
):
    work_dir = f"{work_dir}/training/job_{job_number+1}"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # Logger to record general training information
    logger = create_logger(
        f"subtraining_{job_number+1}",
        os.path.join(work_dir, "messages.txt"),
    )

    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count()={torch.cuda.device_count()}")

    # Logger to record losses against epochs during training
    loss_logger = create_logger(
        f"losses_{job_number+1}",
        os.path.join(work_dir, "losses.csv"),
        simple_fmt=True,
    )

    # Keep CPU thread usage low (often helps on shared/VM environments)
    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)

    # Always run serially on a single process/GPU
    # If CUDA is available, use GPU 0 (or the only visible GPU via CUDA_VISIBLE_DEVICES).
    igpu = 0 if torch.cuda.is_available() else -1

    start = time.time()
    logger.info(f"Training started for trial {job_number+1}.")

    trainer = Trainer.from_data(
        data_file,
        igpu=igpu,
        verbose=verbose,
        work_dir=work_dir,
        config_parameters=train_config,
        logger=logger,
        loss_logger=loss_logger,
    )

    # Timeout (hours -> seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_hours * 3600))

    metrics = trainer.train()
    logger.info(metrics)

    signal.alarm(0)

    time_used = time.time() - start
    logger.info(f"Training finished. Time used: {time_used:.2f}s.\n\n")

    return metrics, time_used


def _suggest_value(trial, name, spec, default_value):
    if spec is None:
        return default_value
    low = spec.get("low", None)
    high = spec.get("high", None)
    if low is None or high is None:
        return default_value
    dtype = str(spec.get("type", "float")).lower()
    step = spec.get("step", None)
    log = bool(spec.get("log", False))
    if dtype == "int":
        value = trial.suggest_int(name, int(low), int(high), step=int(step or 1), log=log)
    else:
        value = trial.suggest_float(name, float(low), float(high), step=step, log=log)
    if name in {"dis_beta", "gen_beta"}:
        value = min(max(float(value), 0.0), 0.999)
    return value


def _build_sampler(optuna_cfg):
    sampler_name = str(optuna_cfg.get("sampler", "TPESampler"))
    seed = optuna_cfg.get("seed", None)
    if sampler_name.lower() == "randomsampler":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def _maybe_regenerate_data(
    *,
    optuna_cfg: dict,
    train_config: Parameters,
    work_dir: str,
    logger: logging.Logger,
) -> str:
    regen = bool(optuna_cfg.get("regenerate_data", False)) if isinstance(optuna_cfg, dict) else False
    if not regen:
        return os.path.join(work_dir, train_config.get("data_file", ""))

    data_file = train_config.get("data_file", "")
    out_name = os.path.basename(str(data_file)) if data_file else "HB_data_LW_npz.csv"
    out_dir = os.path.join(work_dir, "hbgn_data")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(work_dir, "5main.py"),
        "--out_dir",
        out_dir,
        "--out_name",
        out_name,
    ]
    logger.info(f"Regenerating data via 5main: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=work_dir)

    return os.path.join(out_dir, out_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config for training parameter in YAML format",
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        type=str,
        default=".",
        help="Working directory to write the output files",
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    assert os.path.exists(work_dir)

    train_config = Parameters.from_yaml(os.path.join(work_dir, args.config))

    verbose = train_config.get("verbose", False)
    trials = int(train_config.get("trials", 1))
    data_file = os.path.join(work_dir, train_config.get("data_file", None))
    timeout = train_config.get("timeout", 10)
    optuna_cfg = train_config.get("optuna", None)

    # Main logger
    logger = create_logger(
        "Main training:",
        f"{work_dir}/main_process_message.txt",
        append=True,
    )
    logger.info("START")
    logger.info("Running SERIALLY (no ipyparallel).")
    logger.info("Running with 1 process(es).")

    start = time.time()

    if isinstance(optuna_cfg, dict) and optuna_cfg.get("enabled", False):
        search_space = optuna_cfg.get("search_space", {})
        target_params = [
            "alpha_limit",
            "dis_beta",
            "dis_dropout_rate",
            "dis_noise",
            "gen_beta",
            "dropout_rate",
            "lr_base",
            "spec_noise",
            "weight_decay",
            "s_R",
            "s_theta",
            "s_nu",
            "alpha_R",
            "alpha_theta",
            "alpha_nu",
            "R_max",
            "theta_min",
            "R0",
            "theta0",
            "nu0",
        ]
        sampler = _build_sampler(optuna_cfg)
        direction = optuna_cfg.get("direction", "maximize")
        n_trials = int(optuna_cfg.get("n_trials", trials))

        def objective(trial):
            param_updates = {}
            for name in target_params:
                spec = search_space.get(name, None)
                param_updates[name] = _suggest_value(
                    trial, name, spec, train_config.get(name, None)
                )
            tuned_config = Parameters(
                {**train_config.to_dict(), **param_updates}
            )
            trial_data_file = _maybe_regenerate_data(
                optuna_cfg=optuna_cfg,
                train_config=train_config,
                work_dir=work_dir,
                logger=logger,
            )
            trial_work_dir = os.path.join(
                work_dir, "optuna", f"trial_{trial.number + 1}"
            )
            metrics, _ = run_training(
                job_number=0,
                work_dir=trial_work_dir,
                train_config=tuned_config,
                verbose=verbose,
                data_file=trial_data_file,
                timeout_hours=timeout,
                logger=logger,
            )
            combined_metric = (
                np.array(Trainer.metric_weights) * np.array(metrics)
            ).sum()
            trial.set_user_attr("metrics", metrics)
            return combined_metric

        logger.info(
            f"Optuna enabled. Running {n_trials} trial(s) with direction={direction}."
        )
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial
        logger.info(f"Optuna best value: {best.value}")
        logger.info(f"Optuna best params: {best.params}")
        try:
            importances = get_param_importances(study)
            logger.info("Optuna parameter importances (desc):")
            for k, v in importances.items():
                logger.info(f"  {k}: {v:.6f}")
            with open(os.path.join(work_dir, "optuna_importance.yaml"), "w") as f:
                f.write("param_importance:\n")
                for k, v in importances.items():
                    f.write(f"  {k}: {v}\n")
            with open(os.path.join(work_dir, "optuna_importance.csv"), "w") as f:
                f.write("param,importance\n")
                for k, v in importances.items():
                    f.write(f"{k},{v}\n")
        except Exception as e:
            logger.info(f"Optuna importance computation failed: {e}")
        with open(os.path.join(work_dir, "optuna_best.yaml"), "w") as f:
            f.write("best_value: " + str(best.value) + "\n")
            f.write("best_params:\n")
            for k, v in best.params.items():
                f.write(f"  {k}: {v}\n")
        end = time.time()
        logger.info(
            f"Total time used: {end - start:.2f}s for {n_trials} optuna trials."
        )
        logger.info("END\n\n")
        return

    results = []
    for trial in range(trials):
        r = run_training(
            job_number=trial,
            work_dir=work_dir,
            train_config=train_config,
            verbose=verbose,
            data_file=data_file,
            timeout_hours=timeout,
            logger=logger,
        )
        results.append(r)

    time_trials = np.array([r[1] for r in results], dtype=float)
    logger.info(
        f"Time used for each trial: {time_trials.mean():.2f} +/- {time_trials.std():.2f}s.\n"
        + " ".join([f"{t:.2f}s" for t in time_trials])
    )

    end = time.time()
    logger.info(
        f"Total time used: {end - start:.2f}s for {trials} trails "
        f"({(end - start) / max(trials, 1):.2f} each on average)."
    )
    logger.info("END\n\n")


if __name__ == "__main__":
    main()
