# from config import EnvConfig
import os
from typing import Tuple

import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd

# config = EnvConfig()
import yaml


color_list = [
    "indigo",
    "teal",
    "royalblue",
    "m",
    "hotpink",
    "c",
    "r",
    "y",
    "g",
    "b",
    "orange",
    "purple",
    "w",
    "k",
    "tan",
]


def get_yaml_data(yaml_file):
    # 打开yaml文件
    file = open(yaml_file, "r", encoding="utf-8")
    file_data = file.read()
    file.close()
    # 将字符串转化为字典或列表
    data = yaml.safe_load(file_data)
    return data

def plot_aoi_episode(data: pd.DataFrame, save_dir: str):
    # episode中每个slot的各sn的aoi
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    aoi = []
    # data = pd.read_excel(data_file_path, sheet_name=None)
    for id in range(sn_num):
        sn_data = data[f"sn{id}"]
        aoi_i = []

        for i in range(sn_data.shape[0]):
            # fuav_data = fuav_data.iloc[i]
            if sn_data["step"][i] == 1:
                x = sn_data["aoi"][i]
                aoi_i.append(x)
        aoi.append(aoi_i)    
    averages = [sum(values) / len(values) for values in zip(*aoi)] 
    

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(averages, marker='o', linestyle='-', color='b', label="Average AOI")
       
    ax.set_xlabel("slot")
    ax.set_ylabel("aoi")
    ax.set_title("Average AOI Value")
   
    plt.show()
    plt.savefig(f"{save_dir}/aoi1.png")
    print(f"Finish plotting aoi1.")

def plot_aoi_episode_compare(data_path, save_dir: str):
    # episode中每个slot的各sn的aoi
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    algo = ["bkm","kmeans","gmm"]
    
    
    for algo_id in range(len(data_path)):
        aoi = []
        # averages = []
        data = pd.read_excel(data_path[algo_id], sheet_name=None)
        for id in range(sn_num):
            sn_data = data[f"sn{id}"]
            aoi_i = []

            for i in range(sn_data.shape[0]):
                # fuav_data = fuav_data.iloc[i]
                if sn_data["step"][i] == 1:
                    x = sn_data["aoi"][i]
                    aoi_i.append(x)
            aoi.append(aoi_i)    
        averages = [sum(values) / len(values) for values in zip(*aoi)] 
        ax.plot(averages, marker='o', linestyle='-', color=color_list[algo_id], label=algo[algo_id])
    
    # ax.plot(averages, marker='o', linestyle='-', color='b', label="bkm")
    # ax.plot(y2, marker='s', linestyle='--', color='r', label="kmeans")   
    # ax.plot(y3, marker='^', linestyle=':', color='g', label="gmm") 
    ax.set_xlabel("slot")
    ax.set_ylabel("aoi")
    ax.set_title("Average AOI Value")
    
    ax.legend(loc="best")
   
    plt.show()
    plt.savefig(f"{save_dir}/aoi_compare.png")
    print(f"Finish plotting aoi_compare.")

def plot_aoi_max_compare(data_path, save_dir: str):
    # 各sn的最大aoi
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    algo = ["bkm","kmeans","gmm"]
    
    
    for algo_id in range(len(data_path)):
        aoi = []
        data = pd.read_excel(data_path[algo_id], sheet_name=None)
        for id in range(sn_num):
            sn_data = data[f"sn{id}"]
            
            sn_data_f = sn_data.iloc[-1]
            l_aoi = eval(sn_data_f["last_aoi"])
            
            aoi.append(max(l_aoi))
        assert len(aoi) == sn_num
        aoi.sort(reverse=True)     
        ax.plot(aoi, marker='o', linestyle=':', color=color_list[algo_id], label=algo[algo_id])
        
    ax.set_xlabel("sn")
    ax.set_ylabel("max_aoi")
    ax.set_title("Max AOI Value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best")
   
    plt.show()
    plt.savefig(f"{save_dir}/maxaoi_compare.png")
    print(f"Finish plotting max_aoi_compare.")

def plot_data_energy_compare(data_path, save_dir: str):
    # 数据收集量 和 剩余能量
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    algo = ["bkm","kmeans","gmm"]
    X = np.arange(2)
    
    for algo_id in range(len(data_path)):
        d_e = []
        d = 0
        e = 0
        data = pd.read_excel(data_path[algo_id], sheet_name=None)
        for f_id in range(fuav_num):
            f_data = data[f"fuav{f_id}"]
            f_data_f = f_data.iloc[-1]
            # a = float(f_data_f["energy"])
            # e += a
            e += f_data_f["energy"]
             
            for i in range(f_data.shape[0]):
                if len(f_data["sn_data"][i]) >= 1:
                    # d += b
                    # b = float(f_data["dist_or_data"][i])
                    d += f_data["dist_or_data"][i]
        
        
        d_e = [d/fuav_num,e/fuav_num]
        print(algo_id)
        print(d_e)
        p = 0.25* algo_id
        if algo_id == 1:
            ax.bar(X + p,d_e, color = color_list[algo_id], width = 0.25, label=algo[algo_id],tick_label=["data_volume","energy"])            
        else:
            ax.bar(X + p, d_e, color = color_list[algo_id], width = 0.25, label=algo[algo_id])  
        
        
    # ax.set_xlabel("sn")
    # ax.set_ylabel("max_aoi")
    ax.set_title("Data Volume and Energy")
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best")
   
    plt.show()
    plt.savefig(f"{save_dir}/data_energy.png")
    print(f"Finish plotting data_energy_compare.")

def plot_kl_episode_compare(data_path, save_dir: str):
    # episode中每个slot的各uav的kl散度
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    algo = ["greedy","hungarian","sorted"]

    
    
    for algo_id in range(len(data_path)):
        aoi = []
        # averages = []
        data = pd.read_excel(data_path[algo_id], sheet_name=None)
        for id in range(fuav_num):
            fuav_data = data[f"fuav{id}"]
            aoi_i = []

            for i in range(fuav_data.shape[0]):
                # fuav_data = fuav_data.iloc[i]
                if fuav_data["step"][i] == 1:
                    x = fuav_data["aoi"][i]
                    aoi_i.append(x)
            aoi.append(aoi_i)    
        averages = [sum(values) / len(values) for values in zip(*aoi)] 
        ax.plot(averages, marker='o', linestyle='-', color=color_list[algo_id], label=algo[algo_id])
    
    # ax.plot(averages, marker='o', linestyle='-', color='b', label="bkm")
    # ax.plot(y2, marker='s', linestyle='--', color='r', label="kmeans")   
    # ax.plot(y3, marker='^', linestyle=':', color='g', label="gmm") 
    ax.set_xlabel("slot")
    ax.set_ylabel("aoi")
    ax.set_title("Average AOI Value")
    
    ax.legend(loc="best")
   
    plt.show()
    plt.savefig(f"{save_dir}/aoi_compare.png")
    print(f"Finish plotting aoi_compare.")



if __name__ == "__main__":
    data_file_path = [
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/TPRA/cluster/319895/0_bkm/1001.xlsx",
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/TPRA/cluster/319895/1_kmeans/1001.xlsx",
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/TPRA/cluster/319895/2_gmm/1001.xlsx",
    ]
    save_dir_uav = ["/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/record/TPRA/cluster/319895/"]
    yaml_path = [
        "/home/ustc-lc1/ywk/pymarl_uav/multi-agent-comm-net.gi/src/config/envs/SNode.yaml"
    ]

    # for i in range(len(data_file_path)):

    #     cfg = get_yaml_data(yaml_path[i])

    #     sn_num = cfg["env_args"]["sn_num"]
    #     fuav_num = cfg["env_args"]["fuav_num"]
        
        # data_file = pd.read_excel(data_file_path[i], sheet_name=None)

        # if not os.path.exists(save_dir_uav[i]):
        #     os.makedirs(save_dir_uav[i])
        
        # plot_aoi_episode(data_file,save_dir_uav[i])    
    
    cfg = get_yaml_data(yaml_path[0])
    sn_num = cfg["env_args"]["sn_num"]
    fuav_num = cfg["env_args"]["fuav_num"]
    if not os.path.exists(save_dir_uav[0]):
            os.makedirs(save_dir_uav[0])
    plot_aoi_episode_compare(data_file_path,save_dir_uav[0])
    plot_aoi_max_compare(data_file_path,save_dir_uav[0])
    plot_data_energy_compare(data_file_path,save_dir_uav[0])        
            
            
            
        