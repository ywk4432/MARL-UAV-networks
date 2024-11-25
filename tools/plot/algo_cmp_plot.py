import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.envs.env_1.uav import UAV


def read_data(
    env_config: Path,
    ue_status_file: Path,
    uav_status_file: List[Path],
    system_status_file: Path,
) -> Tuple[float, float, float, float]:
    """
    为某种算法读取并计算性能数据
    Args:
        env_config: 环境配置文件路径
        ue_status_file: 保存了用户每个时隙被覆盖的无人机编号和累积被覆盖的时隙数
        uav_status_file: 从中读取无人机的平均能量消耗
        system_status_file: 从中读取平均 Cluster 覆盖成功率和覆盖公平指数
    Returns:
        [平均 UE 覆盖率，平均 Cluster 覆盖成功率，覆盖公平指数，平均无人机能量消耗]
    """
    ue_status = pd.read_csv(ue_status_file)
    slot_num = len(ue_status)
    ave_ue_cover = np.array(list(map(eval, ue_status.iloc[-1])))[:, 1].mean()
    ave_ue_cover /= slot_num
    system_status = pd.read_csv(system_status_file)
    cluster_cover = eval(system_status["cluster_covered"].iloc[-1])
    ave_cluster_cover = np.mean(cluster_cover)
    fairness = system_status["fairness"].iloc[-1]
    uav_energy = [pd.read_csv(file)["slot_energy"].mean() for file in uav_status_file]
    ave_uav_energy = np.mean(uav_energy)
    with open(f"src/config/envs/{env_config}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    uav = UAV(slot_length=config["slot_length"], **config["uav"])
    ave_uav_energy /= uav.max_energy_in_a_slot
    return ave_ue_cover, ave_cluster_cover, fairness, ave_uav_energy


def plot(
    plot_data: Dict[str, Tuple[float, float, float, float]],
    save_path: Path,
    fig_name: str,
) -> None:
    plt.figure(figsize=(6, 3), dpi=200)
    plt.ylabel("Normalized Performance Index")
    plt.grid(axis="y")
    x_title = [
        "$C$",
        "$\\tilde{C}$",
        "$F$",
        "$\\bar{E}$",
    ]
    colors = ["red", "blue", "green", "purple"]
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


def algo_cmp_plot(env_config: str, label_path_csv: str, fig_name: str):
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
        ue_status_file = prefix / "ue" / "ue_status.csv"
        system_status_file = prefix / "system_status.csv"
        plot_data[row["label"]] = read_data(
            Path(env_config), ue_status_file, uav_status_file, system_status_file
        )
    plot(plot_data, Path("fig/algo_cmp/"), fig_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_config", type=str)
    parser.add_argument("label_path_csv", type=str)
    parser.add_argument("output_name", type=str)
    args = parser.parse_args()
    algo_cmp_plot(args.env_config, args.label_path_csv, args.output_name)


if __name__ == "__main__":
    main()
