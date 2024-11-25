# 本脚本需在 master 分支的根目录下执行
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def read_data(
    uav_serve_capacity: int,
    uav_status_file: List[Path],
    system_status_file: Path,
) -> Tuple[float, float, float]:
    """
    为某种算法读取并计算性能数据
    Args:
        uav_serve_capacity: 无人机最多服务的用户数目
        uav_status_file: 从中读取无人机的平均能量消耗
        system_status_file: 从中读取平均 Cluster 覆盖成功率和覆盖公平指数
    Returns:
        [平均 UE 覆盖率，覆盖公平指数，无人机资源利用率]
    """
    system_status = pd.read_csv(system_status_file)
    ave_ue_cover = system_status["ue_cover_rate"].iloc[-1]
    fairness = system_status["fairness"].iloc[-1]
    uav_usage = [
        pd.read_csv(file)["serve_ue_count"].iloc[-1] for file in uav_status_file
    ]
    ave_uav_usage = np.mean(uav_usage) / uav_serve_capacity
    return ave_ue_cover, fairness, ave_uav_usage


def plot(
    plot_data: Dict[str, Tuple[float, float, float]],
    save_path: Path,
    fig_name: str,
) -> None:
    plt.figure(figsize=(6, 4), dpi=200)
    plt.grid(axis="y")
    x_title = ["$S$", "$F$", "$U$"]
    colors = ["red", "blue", "green", "yellow"]
    x = np.arange(len(x_title))
    width = 0.15
    for i, (label, data) in enumerate(plot_data.items()):
        plt.bar(
            x + i * width, data, width=width, label=label, color=colors[i], align="edge"
        )
    plt.xticks(x + ((len(plot_data) + 1) // 2) * width, x_title)
    plt.legend()
    png_path = save_path / "png"
    pdf_path = save_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    plt.savefig(png_path / f"{fig_name}.png", bbox_inches="tight")
    plt.savefig(pdf_path / f"{fig_name}.pdf", bbox_inches="tight")
    plt.close()


def algo_cmp_plot(config: str, label_path_csv: str, fig_name: str):
    with open(f"src/config/envs/{config}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    algos = pd.read_csv(label_path_csv)
    plot_data = dict()
    for _, row in algos.iterrows():
        prefix = Path("record") / row["path"]
        if len(list(prefix.glob("system_status.csv"))) == 0:
            paths = sorted(prefix.iterdir())[:-1]
            last_t_env = sorted(map(lambda x: int(x.name), paths))[-1]
            prefix /= str(last_t_env)
        uav_file_prefix = prefix / "uav"
        uav_status_file = sorted(uav_file_prefix.glob("uav_*.csv"))
        system_status_file = prefix / "system_status.csv"
        plot_data[row["label"]] = read_data(
            config["uav"]["max_serve_capacity"], uav_status_file, system_status_file
        )
        plot(plot_data, Path("fig/algo_cmp/"), fig_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("label_path_csv", type=str)
    parser.add_argument("output_name", type=str)
    args = parser.parse_args()
    algo_cmp_plot(args.config, args.label_path_csv, args.output_name)


if __name__ == "__main__":
    main()
