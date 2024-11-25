import argparse
import math
import os
import subprocess

# uav_num = 3
python = "/home/ustc-lc1/miniconda3/envs/pymarl/bin/python"
run_args = [
    "--env-config=formation_sub1",
    "--env-config=formation_sub2",
    "--env-config=formation_sub3",
    "--env-config=formation_sub4",
]


def get_command(run_arg: str):
    return (
        f"{python} {os.getcwd()}/src/main.py "
        f"--config=two_timescale {run_arg}"
        f"with use_cuda=True"
    )


def main(job_num: int):
    group_num = math.ceil(len(run_args) / job_num)
    groups = [
        run_args[i * job_num : min((i + 1) * job_num, len(run_args))]
        for i in range(group_num)
    ]
    for group in groups:
        processes = [
            subprocess.Popen(get_command(arg).split(), cwd=os.getcwd()) for arg in group
        ]
        for process in processes:
            process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--python", type=str, default=None)
    parser.add_argument("-j", "--job-num", type=int, default=4)
    args = parser.parse_args()
    if args.python is not None:
        python = args.python
    main(args.job_num)
