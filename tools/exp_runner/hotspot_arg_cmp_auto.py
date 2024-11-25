import argparse
import math
import os
import subprocess
from pathlib import Path

import yaml

python = "/home/ustc-lc1/miniconda3/envs/pymarl/bin/python"
arg_name = "max_serve_capacity"
arg_values = {
    "max_serve_capacity": [10, 15, 20, 25, 30],
    "max_obs_radius": [1, 2, 3, 4, 5],
}


def get_command(arg_val: float):
    return (
        f"{python} {os.getcwd()}/src/main.py "
        f"--config=qmix_gat --env-config=hotspot_tmp_{arg_val} "
        f"with run_id=env2_arg_cmp/{arg_name}/{arg_val} use_cuda=True"
    )


def main(job_num: int):
    with open(f"../../src/config/envs/hotspot.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    test_args = arg_values[arg_name]
    for val in test_args:
        config["env_args"]["uav"][arg_name] = val
        with open(
            f"src/config/envs/hotspot_tmp_{val}.yaml", "w", encoding="utf-8"
        ) as f:
            yaml.dump(config, f)
    group_num = math.ceil(len(test_args) / job_num)
    groups = [
        test_args[i * job_num : min((i + 1) * job_num, len(test_args))]
        for i in range(group_num)
    ]
    for group in groups:
        processes = [
            subprocess.Popen(get_command(val).split(), cwd=os.getcwd()) for val in group
        ]
        for process in processes:
            process.wait()
    config_path = Path("../../src/config/envs")
    for path in config_path.glob("hotspot_tmp_*.yaml"):
        path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arg-name", type=str, default=None)
    parser.add_argument("-p", "--python", type=str, default=None)
    parser.add_argument("-j", "--job-num", type=int, default=5)
    args = parser.parse_args()
    if args.python is not None:
        python = args.python
    if args.arg_name is not None:
        arg_name = args.arg_name
    main(args.job_num)
