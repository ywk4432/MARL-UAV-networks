import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from algo_cmp_plot import read_data as get_algo_data

algo_names = ["CEN-CTP", "QMIX", "DMTD", "Greedy"]
performance_label = ["$C$", "$\\tilde{C}$", "$F$", "$\\bar{E}$"]
# performance_label = [
#     "Average UE Coverage",
#     "Average UE Cluster Coverage Success Rate",
#     "Coverage Fairness Index",
#     "Average UAV Energy Consumption",
# ]
performance_names = ["ue_cover_rate", "cluster_cover_rate", "fairness", "slot_energy"]


def get_exp_data(env_config: Path, prefix: Path):
    algos = pd.read_csv(prefix / "algo_name.csv")
    exp_data = dict()  # 一组对比实验的数据
    for _, row in algos.iterrows():
        prefix = Path("record") / row["path"]
        uav_file_prefix = prefix / "uav"
        uav_status_file = sorted(uav_file_prefix.glob("uav_*.csv"))
        ue_status_file = prefix / "ue" / "ue_status.csv"
        system_status_file = prefix / "system_status.csv"
        exp_data[row["label"]] = get_algo_data(
            env_config, ue_status_file, uav_status_file, system_status_file
        )
    return exp_data


def get_data(env_config: Path, arg_csv_path: Path):
    cmp_args = pd.read_csv(arg_csv_path)
    plot_data = dict()
    for _, row in cmp_args.iterrows():
        plot_data[row["value"]] = get_exp_data(
            env_config, Path(f"record/{row['path']}")
        )
    return plot_data


def plot(
    arg_name: str,
    performance_id: int,
    plot_data: Dict[int, Dict[str, Tuple[float, float, float, float]]],
    save_path: Path,
) -> None:
    """
    绘制各种算法在 arg_name 变化时性能指标变化的折线图
    Args:
        arg_name: 变化的变量名
        performance_id: 性能指标在 get_algo_data 返回值的下标
        plot_data: {arg_val: exp_data}
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(6, 4), dpi=200)
    plt.xlabel(arg_name, fontsize=16)
    plt.ylabel(performance_label[performance_id], fontsize=16)
    plt.grid()
    colors = ["red", "blue", "green", "purple"]
    markers = ["o", "v", "^", "*"]
    arg_vals = list(plot_data)
    for i, name in enumerate(algo_names):
        y = [plot_data[val][name][performance_id] for val in arg_vals]
        plt.plot(arg_vals, y, color=colors[i], marker=markers[i], label=name)
    plt.legend(fontsize=14, loc="upper right")
    fig_name = performance_names[performance_id]
    png_path = save_path / "png"
    pdf_path = save_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    plt.savefig(png_path / f"{fig_name}.png", bbox_inches="tight")
    plt.savefig(pdf_path / f"{fig_name}.pdf", bbox_inches="tight")
    plt.close()


def arg_cmp_plot(env_config: str, arg_csv: str, arg_name: str):
    plot_data = get_data(Path(env_config), Path(arg_csv))
    for i in range(len(performance_label)):
        plot(arg_name, i, plot_data, Path(f"fig/arg_cmp/{arg_name}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_config", type=str)
    parser.add_argument("arg_csv", type=str)
    parser.add_argument("arg_name", type=str)
    args = parser.parse_args()
    arg_cmp_plot(args.env_config, args.arg_csv, args.arg_name)


if __name__ == "__main__":
    main()
