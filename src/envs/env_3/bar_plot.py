import math
from pathlib import Path
from typing import Dict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# title = ['HRL-T^2','IAPF','A*-DWA','DMTD']
alive = [1, 10 / 14, 7 / 14, 2 / 14]


def get_yaml_data(yaml_file):
    # 打开yaml文件
    file = open(yaml_file, "r", encoding="utf-8")
    file_data = file.read()
    file.close()
    # 将字符串转化为字典或列表
    data = yaml.safe_load(file_data)
    return data


def read_data(
    methed_i, yaml_path, file_path, slot
) -> Tuple[float, float, float, float]:
    """
    Returns:
        [覆盖率，路径长度/50，存活率，编队保持因子]
    """
    cfg = get_yaml_data(yaml_path)
    luav_num = cfg["env_args"]["luav_num"]
    fuav_num = cfg["env_args"]["fuav_num"]
    slot_step_num = cfg["env_args"]["slot_step_num"]
    luav_pos = np.array(cfg["env_args"]["luav_init_pos_list"][0])
    luav_pos = luav_pos[:2]

    data = pd.read_excel(file_path, sheet_name=None)

    # 覆盖率？
    cover_rate = 1

    # 路径长度
    path_length = 0
    for luav_id in range(luav_num):
        luav_data = data[f"luav{luav_id}"]
        last_pos = luav_pos
        for i in range(luav_data.shape[0]):
            pos = eval(luav_data["pos"][i])[:2]
            dis = math.sqrt((last_pos[0] - pos[0]) ** 2 + (last_pos[1] - pos[1]) ** 2)
            path_length += dis
            last_pos = pos
    print(path_length)
    # 存活率：直接给
    alive_rate = alive[methed_i]
    # 编队保持因子**
    formation_factor = 1
    formation = np.full((fuav_num, slot), -1)
    for fuav_id in range(fuav_num):
        fuav_data = data[f"fuav{fuav_id}"]
        slot_c = 0
        for i in range(1, slot * slot_step_num + 1):
            pos_rela = eval(fuav_data["pos_rela"][i])
            init_rela = eval(fuav_data["init_pos_rela"][i])
            if i % slot_step_num == 0:
                # print(i)
                # print(slot_c)

                dis_rela = math.sqrt(
                    (init_rela[0] - pos_rela[0]) ** 2
                    + (init_rela[1] - pos_rela[1]) ** 2
                )
                formation[fuav_id][slot_c] = dis_rela
                slot_c += 1
            if not fuav_data["formation"][i]:
                break
    # print(formation)
    formation_factor_t = np.zeros(slot)
    for t in range(slot):

        add_t = 0
        len_t = 0
        for j in range(fuav_num):
            if formation[j][t] != -1:
                add_t += formation[j][t]
                len_t += 1
                # print(len_t)
                # print(add_t)

        if add_t == 0:
            formation_factor_t[t] = 0
            # print("addt == 0")
        else:
            formation_factor_t[t] = add_t / len_t
            # print(f"else  t:{t}")
            # print(add_t)
            # print(len_t)
            # print(formation_factor_t[t])
            # print(formation_factor_t)

    print(formation_factor_t)
    formation_factor = sum(formation_factor_t) / slot
    print(formation_factor)
    print([cover_rate, path_length, alive_rate, formation_factor])
    return [path_length, alive_rate, formation_factor]


def normalization(
    plot_data: Dict[str, Tuple[float, float, float, float]],
):
    # 归一化
    m = len(plot_data)
    l = 3
    print(l)
    for i in range(l):
        max = 0
        for j in plot_data:
            if plot_data[j][i] > max:
                max = plot_data[j][i]
        print(max)
        for j in plot_data:
            plot_data[j][i] = plot_data[j][i] / max

    title = ["HRL-T^2", "IAPF", "A*-DWA", "DMTD"]
    plot_data_nor = dict()
    for i in plot_data:
        plot_data_nor[i] = plot_data[i]
    print(plot_data_nor)
    return plot_data_nor


def plot(
    plot_data: Dict[str, Tuple[float, float, float, float]],
    save_path: Path,
) -> None:
    # plot_data = normalization(plot_data)
    print(plot_data)
    plt.figure(figsize=(6, 3), dpi=200)
    plt.ylabel("Normalized Performance Index")
    plt.grid(axis="y")
    x_title = [
        "$f_L$",
        "$f_A$",
        "$\\bar{f_K}$",
    ]
    colors = ["red", "blue", "green", "purple"]
    x = np.arange(len(x_title))
    width = 0.15
    for i, (label, data) in enumerate(plot_data.items()):
        plt.bar(
            x + i * width, data, width=width, label=label, color=colors[i], align="edge"
        )
    plt.xticks(x + ((len(plot_data) + 1) // 2) * width, x_title)
    plt.legend(fontsize=7)
    png_path = save_path / "png"
    pdf_path = save_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    fig_name = "bar"
    plt.savefig(png_path / f"{fig_name}.png", bbox_inches="tight")
    plt.savefig(pdf_path / f"{fig_name}.pdf", bbox_inches="tight")
    plt.close()


def main():
    plot_data = dict()
    # 四个算法的config和file路径
    title = ["HRL-T^2", "IAPF", "A*-DWA", "DMTD"]
    yaml_path = [
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation.yaml",
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation_apf.yaml",
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation_A*.yaml",
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation.yaml",
    ]
    file_path = [
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/1310237/data/2952432.xlsx",
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/apf/397272/91.xlsx",
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/A*/2520827/121.xlsx",
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/202300/data/2852352.xlsx",
    ]
    slot_num = [9, 47, 10, 9]
    for i in range(4):

        plot_data[title[i]] = read_data(i, yaml_path[i], file_path[i], slot_num[i])

    plot(plot_data, Path("figure/bar_plot/"))


if __name__ == "__main__":
    main()
