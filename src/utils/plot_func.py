import csv
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np


def reward_plot(reward_list, name, batch_size_run):
    min_length = len(reward_list[0])
    reward_max_list = []
    reward_min_list = []
    reward_ave_list = []
    reward_idx_list = []

    for idx in range(min_length):
        reward_max = 0
        reward_min = 0
        reward_ave = 0
        reward_idx = []
        for env_idx in range(batch_size_run):
            reward_idx.append(reward_list[env_idx][idx])
        # reward_max_list.append(np.max(reward_idx))
        # reward_min_list.append(np.min(reward_idx))
        reward_ave_list.append(np.mean(reward_idx))
        reward_idx_list.append(idx)

    plt.clf()
    plt.plot(reward_idx_list, reward_ave_list, label='Mean')
    # plt.fill_between(reward_idx_list, reward_min_list, reward_max_list, alpha=0.2, label='Reward Band')
    
    plt.xlabel('slot')
    plt.ylabel('reward_total')
    plt.title('Reward Band Plot')
    plt.legend()

    path = f"{name}/reward_total"
    if not os.path.exists(path):
        os.makedirs(path)
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(path + f"/{cur_time}.pdf")


def write_csv(path, data_row):
    with open(path, "a+") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)
