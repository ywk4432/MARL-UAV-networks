import argparse
import math
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from greedy import main as greedy
from tools.plot.env2_algo_cmp_plot import algo_cmp_plot
from tools.plot.path_plot import plot as path_plot
from tools.plot.td_loss_plot import main as td_loss_plot

python = "/home/ustc-lc1/miniconda3/envs/pymarl/bin/python"
run_args = [
    [
        "qmix_gatenc",
        "qmix_gat",
        " ",
    ],
    [
        "dmtd",
        "multi_agent_transfer_mlp1",
        "learner=local_learner n_agents=5 uav_num=5 use_novelty=False use_hybrid_novelty=False ",
    ],
    [
        "dmtd_gatenc",
        "qmix_gat",
        "learner=local_learner use_novelty=False use_hybrid_novelty=False ",
    ],
]
algo_names = [
    ["QMIX-GATENC", f"env2_algo_cmp/qmix_gatenc"],
    ["DMTD", f"env2_algo_cmp/dmtd"],
    ["DMTD-GATENC", f"env2_algo_cmp/dmtd_gatenc"],
    ["Greedy", f"env2_algo_cmp/greedy"],
]


def get_command(algo_name: str, algo_config: str, run_arg: str):
    return (
        f"{python} {os.getcwd()}/src/main.py "
        f"--config={algo_config} --env-config=hotspot "
        f"with run_id=env2_algo_cmp/{algo_name} {run_arg} use_cuda=True"
    )


def main(job_num: int):
    td_loss_file_paths = [f"record/{path}" for _, path in algo_names[:-1]]
    for path in td_loss_file_paths:
        path = Path(path) / "td_error_abs_agent.csv"
        if path.exists():
            path.unlink()
    group_num = math.ceil(len(run_args) / job_num)
    groups = [
        run_args[i * job_num : min((i + 1) * job_num, len(run_args))]
        for i in range(group_num)
    ]
    for group in groups:
        processes = [
            subprocess.Popen(get_command(name, config, arg).split(), cwd=os.getcwd())
            for name, config, arg in group
        ]
        for process in processes:
            process.wait()
    greedy(
        "hotspot",
        "env2",
        f"record/env2_algo_cmp/greedy",
    )
    pd.DataFrame(algo_names, columns=["label", "path"]).to_csv(
        f"record/env2_algo_cmp/algo_name.csv", index=False
    )
    algo_cmp_plot("hotspot", f"record/env2_algo_cmp/algo_name.csv", f"env2_algo_cmp")
    with open(f"../../src/config/envs/hotspot.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    for name, path in algo_names:
        path_plot(
            config,
            Path("record") / path,
            name,
            Path("fig/episode_path/env2_algo_cmp"),
            config["map_height"],
        )
    td_loss_labels = [name for name, _ in algo_names[:-1]]
    td_loss_plot(
        td_loss_file_paths,
        1,
        td_loss_labels,
        20,
        1,
        "env2_algo_cmp",
        80000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--python", type=str, default=None)
    parser.add_argument("-j", "--job-num", type=int, default=3)
    args = parser.parse_args()
    if args.python is not None:
        python = args.python
    main(args.job_num)
