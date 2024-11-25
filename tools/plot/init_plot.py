import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

dense = 1  # 格子内超过 1 人的为红色，否则为灰色


def plot(config: dict) -> None:
    """
    绘制环境的初始状态
    :param config: 环境配置文件
    """
    plt.figure(dpi=200)
    plt.grid()
    plt.xlim(0, config["map_length"])
    plt.ylim(0, config["map_width"])
    uav_init_pos = np.array(config["uav"]["initial_pos"]) + 0.5
    ue_pos = pd.read_csv(config["ue_pos_file"]).values[:, -2:]
    offset = np.random.uniform(low=0, high=0.5, size=ue_pos.shape)
    ue_pos = ue_pos.astype(np.float64) + offset
    plt.scatter(ue_pos[:, 0], ue_pos[:, 1], color="blue", s=20, label="UE", alpha=0.5)
    plt.scatter(
        uav_init_pos[:, 0], uav_init_pos[:, 1], color="green", marker="*", label="UAV"
    )
    save_path = Path(f"fig/init/")
    png_path = save_path / "png"
    pdf_path = save_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    plt.legend(fontsize=13)
    plt.savefig(png_path / f"{config['map_name']}.png", bbox_inches="tight")
    plt.savefig(pdf_path / f"{config['map_name']}.pdf", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    with open(f"src/config/envs/{args.config}.yaml", "r", encoding="utf-8") as f:
        config: dict = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    plot(config)


if __name__ == "__main__":
    main()
