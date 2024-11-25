# 本脚本需在 master 分支的根目录下执行
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

from env2_algo_cmp_plot import read_data as get_performance_data

cmp_args = {
    "max_obs_radius": {"value": [1, 2, 3, 4, 5], "notation": "$r_{obs}$"},
    "max_serve_capacity": {"value": [5, 10, 15, 20, 25, 30], "notation": "$c_n^*$"},
}
performance_label = ["$S$", "$F$", "$U$"]


def get_data(config: dict, prefix: Path):
    res = dict()
    for arg_name, arg_values in cmp_args.items():
        data = []
        for arg_value in arg_values["value"]:
            uav_serve_capacity = (
                arg_value
                if arg_name == "max_serve_capacity"
                else config["uav"]["max_serve_capacity"]
            )
            exp_prefix = prefix / arg_name / str(arg_value)
            uav_file_prefix = exp_prefix / "uav"
            uav_status_file = sorted(uav_file_prefix.glob("uav_*.csv"))
            system_status_file = exp_prefix / "system_status.csv"
            data.append(
                (
                    arg_value,
                    get_performance_data(
                        uav_serve_capacity, uav_status_file, system_status_file
                    ),
                )
            )
        res[arg_name] = data
    return res


def plot(
    arg_name: str,
    plot_data: Dict[str, List[Tuple[float, Tuple[float, float, float]]]],
    save_path: Path,
) -> None:
    plt.figure(figsize=(6, 4), dpi=200)
    plt.xlabel(cmp_args[arg_name]["notation"])
    plt.grid()
    colors = ["red", "blue", "green"]
    markers = ["o", "^", "*"]
    arg_vals = cmp_args[arg_name]["value"]
    y_data = np.array([data for _, data in plot_data[arg_name]])
    for i, label in enumerate(performance_label):
        plt.plot(
            arg_vals, y_data[:, i], color=colors[i], marker=markers[i], label=label
        )
    plt.legend()
    fig_name = arg_name
    png_path = save_path / "png"
    pdf_path = save_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    plt.savefig(png_path / f"{fig_name}.png", bbox_inches="tight")
    plt.savefig(pdf_path / f"{fig_name}.pdf", bbox_inches="tight")
    plt.close()


def arg_cmp_plot(env_config: str, run_id: str):
    with open(f"src/config/envs/{env_config}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    plot_data = get_data(config, Path(f"record/{run_id}"))
    for name in cmp_args:
        plot(name, plot_data, Path(f"fig/arg_cmp/{run_id}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_config", type=str)
    parser.add_argument("run_id", type=str)
    args = parser.parse_args()
    arg_cmp_plot(args.env_config, args.run_id)


if __name__ == "__main__":
    main()
