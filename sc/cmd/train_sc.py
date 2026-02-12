import argparse
import logging
import os
import signal
import time

import numpy as np
import torch

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
