import argparse
import os
import subprocess
from datetime import datetime

import pandas as pd

run_args = [
    "model_type=ue use_cuda=False",
    "model_type=energy use_cuda=True",
    "model_type=illegal use_cuda=True",
    "model_type=default model_save=False use_cuda=True",
    "model_load=True model_save=False use_cuda=False",
]


def main(command: str, run_id: str, extra_args: str):
    processes = [
        subprocess.Popen(
            f"{command} {arg} {extra_args}".split(),
            cwd=os.getcwd(),
        )
        for arg in run_args[:-1]
    ]
    pids = [process.pid for process in processes]
    for process in processes:
        process.wait()
    load = subprocess.Popen(
        f"{command} {run_args[-1]} {extra_args}".split(),
        cwd=os.getcwd(),
    )
    pids.append(load.pid)
    data = dict(model_type=["ue", "energy", "illegal", "default", "load"], pid=pids)
    pd.DataFrame(data).to_csv(f"record/{run_id}/pids.csv", index=False)
    with open(f"record/{run_id}/extra_args.txt", "w") as file:
        file.write(args.extra_args)
    load.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run-id", type=str, default=datetime.now().strftime("%m_%d_%H_%M_%S")
    )
    parser.add_argument(
        "-p",
        "--python",
        type=str,
        default="/home/ustc-lc1/miniconda3/envs/pymarl/bin/python",
    )
    parser.add_argument("--config", type=str, default="transfer_mlp1")
    parser.add_argument("--env-config", type=str, default="one_cluster")
    parser.add_argument("-e", "--extra-args", type=str, default="")
    args = parser.parse_args()
    run_command = (
        f"{args.python} {os.getcwd()}/src/main.py "
        f"--config={args.config} --env-config={args.env_config} "
        f"with run_id={args.run_id}"
    )
    main(run_command, args.run_id, args.extra_args)
