# from config import EnvConfig
import os
from typing import Tuple

import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# config = EnvConfig()
import yaml

uav_color_list = [
    "y",
    "g",
    "teal",
    "m",
    "hotpink",
    "c",
    "b",
    "r",
    "orange",
    "purple",
    "indigo",
    "tan",
    "royalblue",
    "w",
    "k",
]


def get_yaml_data(yaml_file):
    # 打开yaml文件
    file = open(yaml_file, "r", encoding="utf-8")
    file_data = file.read()
    file.close()
    # 将字符串转化为字典或列表
    data = yaml.safe_load(file_data)
    return data


def map_plot(env_map: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """
    用于绘制 UE 和障碍物
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(0, map_length)
    y = np.arange(0, map_width)
    x, y = np.meshgrid(x, y)
    c = [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0)] + np.linspace(
        (0.9, 0.9, 0.9), (0.0, 0.0, 0.5), max_ue_num_in_a_cell
    ).tolist()
    color_map = colors.LinearSegmentedColormap.from_list("color_map", c, N=256)
    ax.pcolormesh(x, y, env_map, cmap=color_map, vmin=-1, vmax=max_ue_num_in_a_cell)
    return fig, ax


def plot_static(ue_pos: np.ndarray, obs_list: np.ndarray, luav_pos):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.set_xlim(0, map_length)

    ax.set_ylim(0, map_width)
    ax.set_aspect("equal")

    # plt.scatter(ue_pos[:, 0], ue_pos[:, 1], c="gray", s=10, label="UE")

    # ax.scatter(ue_pos[0] + 0.5, ue_pos[1] + 0.5, c="gray", s=10, label="UE")
    # ax.scatter(luav_pos[0] + 0.5, luav_pos[1] + 0.5, c="gray", s=10)

    # cca ,ccb = ue_pos[0] + 0.5, ue_pos[1] + 0.5
    cca, ccb = 42.5, 7.5
    r = 5.5
    # 绘制圆圈
    circle = plt.Circle((cca, ccb), r, color="red", fill=False, linestyle="--")

    # 将圆圈添加到图中
    ax.add_patch(circle)

    for pos, a, b in obs_list:
        x, y = pos[0], pos[1]
        points = [[x, y], [x + a, y], [x + a, y + b], [x, y + b], [x, y]]
        points = np.array(points).transpose()
        ax.fill(points[0], points[1], c="black", linestyle="-", linewidth=0.3)

    return fig, ax


def uav_plot(data: pd.DataFrame, frame_idx: int, ax: plt.Axes, save_dir_uav) -> None:
    for luav_id in range(luav_num):
        luav_data = data[f"luav{luav_id}"]
        if frame_idx == 0:
            frame_idx_l = 0
        else:
            frame_idx_l = (frame_idx + 14) // slot_step_num

        x, y = eval(luav_data["pos"][frame_idx_l])[:2]

        ax.scatter([x + 0.5], [y + 0.5], s=50, c=uav_color_list[luav_id], marker="*")
        # fuav_list = eval(luav_data['fuav_list'][frame_idx_l])
        x, y, c = [], [], []
        for fuav in range(fuav_num):
            fuav_data = data[f"fuav{fuav}"]
            if frame_idx < fuav_data.shape[0]:
                fuav_data = fuav_data.iloc[frame_idx]
                pos = eval(fuav_data["pos_abs"])[:2]
                color = uav_color_list[luav_id] if fuav_data["alive"] else "grey"
                x.append(pos[0] + 0.5)
                y.append(pos[1] + 0.5)
                c.append(color)
        ax.scatter(x, y, c=c, s=10, marker="^")
    plt.savefig(f"{save_dir_uav}/frame{i}.png")
    print(f"Finish plotting frame {i}.")


def plot(data_file_path: str, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_file = pd.read_excel(data_file_path, sheet_name=None)
    data_map = data_file["map"].to_numpy()
    for i in range(data_file["luav0"].shape[0]):
        fig, ax = map_plot(data_map)
        uav_plot(data_file, i, ax)
        fig.savefig(f"{save_dir}/frame{i}.png")
        plt.close(fig)
        print(f"Finish plotting frame {i}.")


def trajectory(data: pd.DataFrame, ax: plt.Axes, save_dir_uav):
    if not os.path.exists(save_dir_uav):
        os.makedirs(save_dir_uav)

    for luav_id in range(luav_num):
        x, y = [], []
        luav_data = data[f"luav{luav_id}"]
        for i in range(luav_data.shape[0]):
            pos = eval(luav_data["pos"][i])[:2]
            color = uav_color_list[-1]
            x.append(pos[0] + 0.5)
            y.append(pos[1] + 0.5)
            # c.append(color)
        ax.plot(x, y, color=color, linestyle="-", linewidth=1)

    for fuav_id in range(fuav_num):
        x, y, c = [], [], []
        fuav_data = data[f"fuav{fuav_id}"]

        for i in range(fuav_data.shape[0]):
            # fuav_data = fuav_data.iloc[i]
            pos = eval(fuav_data["pos_abs"][i])[:2]
            color = uav_color_list[fuav_id] if fuav_data["alive"][i] else "grey"
            x.append(pos[0] + 0.5)
            y.append(pos[1] + 0.5)
            # c.append(color)
        ax.plot(x, y, color=color, linestyle=":", linewidth=0.5)

    luav_data_f = luav_data.iloc[-1]
    x, y = eval(luav_data_f["pos"])[:2]

    color = uav_color_list[-1]

    ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="10")

    for fuav in range(fuav_num):
        fuav_data = data[f"fuav{fuav}"]
        fuav_data = fuav_data.iloc[-1]
        x, y = eval(fuav_data["pos_abs"])[:2]
        color = uav_color_list[fuav]
        ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

    luav_data_s = luav_data.iloc[0]
    x, y = eval(luav_data_s["pos"])[:2]

    color = uav_color_list[-1]

    ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="10")

    for fuav in range(fuav_num):
        fuav_data = data[f"fuav{fuav}"]
        fuav_data = fuav_data.iloc[0]
        x, y = eval(fuav_data["pos_abs"])[:2]
        color = uav_color_list[fuav]
        ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

    plt.savefig(f"{save_dir_uav}/trajectory.pdf")
    print(f"Finish plotting trajectory.")


def trajectory_alive(data: pd.DataFrame, ax: plt.Axes, save_dir_uav):
    if not os.path.exists(save_dir_uav):
        os.makedirs(save_dir_uav)

    for luav_id in range(luav_num):
        x, y = [], []
        luav_data = data[f"luav{luav_id}"]
        for i in range(luav_data.shape[0]):
            pos = eval(luav_data["pos"][i])[:2]
            color = uav_color_list[-1]
            x.append(pos[0] + 0.5)
            y.append(pos[1] + 0.5)
            # c.append(color)
        ax.plot(x, y, color=color, linestyle="-", linewidth=1)

    fuav_alive = np.ones(fuav_num)

    for fuav_id in range(fuav_num):
        x, y, c = [], [], []
        fuav_data = data[f"fuav{fuav_id}"]

        for i in range(fuav_data.shape[0]):
            # fuav_data = fuav_data.iloc[i]
            pos = eval(fuav_data["pos_abs"][i])[:2]
            color = uav_color_list[fuav_id] if fuav_data["formation"][i] else "grey"
            x.append(pos[0] + 0.5)
            y.append(pos[1] + 0.5)
            if not fuav_data["formation"][i]:
                fuav_alive[fuav_id] = 0
                color = uav_color_list[fuav_id]
                ax.plot([pos[0] + 0.5], [pos[1] + 0.5], c=color, marker="^")
                x_min, x_max = round(pos[0] - fobs), round(pos[0] + fobs + 1)
                y_min, y_max = round(pos[1] - fobs), round(pos[1] + fobs + 1)
                points = [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                    [x_min, y_min],
                ]
                points = np.array(points).transpose()
                # plt.fill(points[0], points[1], color="green", alpha=0.2,linewidth=0)
                circle = plt.Circle(
                    (pos[0] + 0.5, pos[1] + 0.5),
                    1.5,
                    color="green",
                    alpha=0.2,
                    linewidth=0,
                )
                ax.add_patch(circle)
                break
        ax.plot(x, y, color=color, linestyle=":", linewidth=0.5)

    luav_data_f = luav_data.iloc[-1]
    x, y = eval(luav_data_f["pos"])[:2]

    color = uav_color_list[-1]

    ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="12")

    alive_num = 0

    for fuav in range(fuav_num):
        if fuav_alive[fuav]:
            alive_num += 1
            fuav_data = data[f"fuav{fuav}"]
            fuav_data = fuav_data.iloc[-1]
            x, y = eval(fuav_data["pos_abs"])[:2]
            color = uav_color_list[fuav]
            ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

            x_min, x_max = round(x - fobs), round(x + fobs + 1)
            y_min, y_max = round(y - fobs), round(y + fobs + 1)
            points = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
            points = np.array(points).transpose()
            # plt.fill(points[0], points[1], color="green", alpha=0.2,linewidth=0)
            circle = plt.Circle(
                (x + 0.5, y + 0.5), 1.5, color="green", alpha=0.2, linewidth=0
            )
            ax.add_patch(circle)

    print(alive_num)
    print(alive_num / len(fuav_alive))

    luav_data_s = luav_data.iloc[0]
    x, y = eval(luav_data_s["pos"])[:2]
    color = uav_color_list[-1]
    ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="12")

    for fuav in range(fuav_num):
        fuav_data = data[f"fuav{fuav}"]
        fuav_data = fuav_data.iloc[0]
        x, y = eval(fuav_data["pos_abs"])[:2]
        color = uav_color_list[fuav]
        ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

    # 画中间过程
    # select = 4
    # luav_data_m = luav_data.iloc[select]
    # x, y = eval(luav_data_m["pos"])[:2]
    # color = uav_color_list[-1]
    # ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="12")

    # for fuav in range(fuav_num):
    #     fuav_data = data[f"fuav{fuav}"]
    #     fuav_data = fuav_data.iloc[select*slot_step_num]
    #     x, y = eval(fuav_data["pos_abs"])[:2]
    #     color = uav_color_list[fuav]
    #     ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

    LUAV = mlines.Line2D(
        [],
        [],
        color=uav_color_list[-1],
        marker="*",
        markersize="9",
        linewidth=1,
        label="LUAV",
    )
    FUAV = mlines.Line2D(
        [],
        [],
        color=uav_color_list[0],
        marker="^",
        markersize="6",
        linewidth=1,
        linestyle=":",
        label="FUAV",
    )

    ax.legend(handles=[LUAV, FUAV], loc="upper right", framealpha=1, fontsize=9)
    # ax.legend(handles=[LUAV],loc='upper left')
    # ax.legend(handles=[FUAV],loc='upper left')

    plt.savefig(f"{save_dir_uav}/trajectory_alive.pdf", bbox_inches="tight")

    print(f"Finish plotting alive trajectory.")


def plot_trajectory(data_file_path, save_dir):
    data_file = pd.read_excel(data_file_path, sheet_name=None)
    fig, ax = plot_static(ue_pos, obstacle_list)
    trajectory(data_file, ax, save_dir)
    plt.close(fig)


if __name__ == "__main__":

    # data_file_path = ["/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/1310237/data/2952432.xlsx",
    #                   "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/52108/data/1351152.xlsx",
    #            "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/apf/397272/91.xlsx",
    #            "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/A*/2520827/121.xlsx",
    #            "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/202300/data/2852352.xlsx"]
    # save_dir_uav = ["figure/formation/1310237/data/2952432/uav",
    #                 "figure/icm/52108/data/1351152/uav",
    #                 "figure/apf/397272/91/uav",
    #                 "figure/A*/2520827/121/uav",
    #                 "figure/DMTD/202300/data/2852352/uav",
    #                 ]
    # yaml_path = ["/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation.yaml",
    #             "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation.yaml",
    #            "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation_apf.yaml",
    #            "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation_A*.yaml",
    #            "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation.yaml"]

    # data_file_path = ["/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/2765249/data/3000000.xlsx",
    #                   "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/160_100_uav_data.xlsx",
    #            "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/1443197/data/1451232.xlsx",
    #            ]
    # save_dir_uav = ["figure/sub1/2765249/data/3000000/uav",
    #                 "figure/sub2/160/data/1000000/uav",
    #                 "figure/sub4/1443197/data/1451232/uav",
    #                 ]
    # yaml_path = ["/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation_sub1.yaml",
    #             "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation_sub2.yaml",
    #            "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation_sub4.yaml",
    #            ]

    data_file_path = [
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/formation/4025466/data/2952432.xlsx"
    ]
    save_dir_uav = ["figure/formation/4025466/data/2952432/uav"]
    yaml_path = [
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/formation.yaml"
    ]

    for i in range(len(data_file_path)):

        cfg = get_yaml_data(yaml_path[i])

        luav_num = cfg["env_args"]["luav_num"]
        fuav_num = cfg["env_args"]["fuav_num"]
        max_ue_num_in_a_cell = cfg["env_args"]["max_ue_num_in_a_cell"]
        slot_step_num = cfg["env_args"]["slot_step_num"]
        map_length = cfg["env_args"]["map_length"]
        map_width = cfg["env_args"]["map_width"]
        fobs = cfg["env_args"]["fuav_observation_size"]
        obstacle_list = np.array(cfg["env_args"]["obstacle_list"], dtype=object)
        ue_pos = np.array(cfg["env_args"]["ue_cluster_center_list"][0], dtype=object)
        ue_pos = ue_pos[0]
        luav_pos = np.array(cfg["env_args"]["luav_init_pos_list"][0])
        luav_pos = luav_pos[:2]

        data_file = pd.read_excel(data_file_path[i], sheet_name=None)
        data_map = data_file["map"].to_numpy()

        if not os.path.exists(save_dir_uav[i]):
            os.makedirs(save_dir_uav[i])
        fig, ax = plot_static(ue_pos, obstacle_list, luav_pos)
        trajectory_alive(data_file, ax, save_dir_uav[i])
        plt.close(fig)

    # if not os.path.exists(save_dir_uav):
    #     os.makedirs(save_dir_uav)
    # for i in range((data_file['luav0'].shape[0]-1) * slot_step_num+1):
    #     fig, ax = plot_static(ue_pos, obstacle_list, luav_pos)
    #     uav_plot(data_file, i, ax, save_dir_uav)
    #     plt.close(fig)

    # fig, ax = plot_static(ue_pos, obstacle_list, luav_pos)
    # plt.savefig(f'figure/map/formation.pdf')
    # print(f'Finish plotting map.')
