import argparse
import math
import os
import subprocess

import pandas as pd

from greedy import main as greedy
from tools.plot.algo_cmp_plot import algo_cmp_plot

uav_num = 3
python = "/home/ustc-lc1/miniconda3/envs/pymarl/bin/python"
run_args = [
    ["qmix_mcne", "learner=q_learner use_novelty=True use_hybrid_novelty=True "],
    ["qmix", "learner=q_learner use_novelty=False use_hybrid_novelty=False "],
    ["dmtd", "learner=local_learner use_novelty=True use_hybrid_novelty=False "],
]


def get_command(algo_name: str, run_arg: str):
    return (
        f"{python} {os.getcwd()}/src/main.py "
        f"--config=multi_agent_transfer_mlp1 --env-config=large_map "
        f"with uav_num={uav_num} "
        f"run_id={uav_num}_uav/{algo_name} "
        f"ue_pos_file=src/envs/env_1/ue_pos_config/{uav_num}_ue_pos.csv "
        f"ue_cluster_file=src/envs/env_1/cluster_config/{uav_num}_cluster_centers.csv "
        f"obs_list_file=src/envs/env_1/obstacle_config/obs_list_34578.csv {run_arg}"
    )


def reconfig(config: dict):
    config.update(
        {
            "uav_num": uav_num,
            "ue_pos_file": f"src/envs/env_1/ue_pos_config/{uav_num}_ue_pos.csv",
            "ue_cluster_file": f"src/envs/env_1/cluster_config/{uav_num}_cluster_centers.csv",
            "obs_list_file": f"src/envs/env_1/obstacle_config/obs_list_34578.csv",
        }
    )


def main(job_num: int):
    group_num = math.ceil(len(run_args) / job_num)
    groups = [
        run_args[i * job_num : min((i + 1) * job_num, len(run_args))]
        for i in range(group_num)
    ]
    for group in groups:
        processes = [
            subprocess.Popen(get_command(name, arg).split(), cwd=os.getcwd())
            for name, arg in group
        ]
        for process in processes:
            process.wait()
    greedy(
        "large_map",
        "env1",
        f"record/algo_cmp/{uav_num}_uav/greedy",
        reconfig,
    )
    algo_names = [
        ["QMIX-MCNE", f"algo_cmp/{uav_num}_uav/qmix_mcne"],
        ["QMIX", f"algo_cmp/{uav_num}_uav/qmix"],
        ["DMTD", f"algo_cmp/{uav_num}_uav/dmtd"],
        ["Greedy", f"algo_cmp/{uav_num}_uav/greedy"],
    ]
    pd.DataFrame(algo_names, columns=["label", "path"]).to_csv(
        f"record/algo_cmp/{uav_num}_uav/algo_name.csv", index=False
    )
    algo_cmp_plot(
        "large_map",
        f"record/algo_cmp/{uav_num}_uav/algo_name.csv",
        f"{uav_num}_uav",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--uav-num", type=int, default=None)
    parser.add_argument("-p", "--python", type=str, default=None)
    parser.add_argument("-j", "--job-num", type=int, default=4)
    args = parser.parse_args()
    if args.python is not None:
        python = args.python
    if args.uav_num is not None:
        uav_num = args.uav_num
    main(args.job_num)
