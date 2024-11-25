"""
 # @ Author: Wenke
 # @ Create Time: 2023-09-18 15:55:19
 # @ Modified by: Wenke
 # @ Modified time: 2023-09-19 06:47:29
 # @ Description: 数据保存功能
 """

import os

import numpy as np
import pandas as pd


def check_file_exists(file_path):
    return os.path.isfile(file_path)


def save_data_to_excel(env, excel_file):
    # data_map = np.zeros((env.map_length, env.map_width))
    # for i in range(env.map_length):
    #     for j in range(env.map_width):
    #         if env.cell_map[i][j].ue_num > 0:
    #             data_map[i][j] = env.cell_map[i][j].ue_num
    #         if env.cell_map[i][j].obs:
    #             data_map[i][j] = -1
    if check_file_exists(excel_file):
        with pd.ExcelFile(excel_file, engine="openpyxl") as xls:
            with pd.ExcelWriter(
                excel_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
            ) as writer:
                for drone_name, drone_df in env.uav_data.items():
                    if drone_name in xls.sheet_names:
                        original_data = pd.read_excel(xls, sheet_name=drone_name)
                        updated_data = pd.concat(
                            [original_data, drone_df], ignore_index=True
                        )
                        updated_data.to_excel(
                            writer, sheet_name=drone_name, index=False
                        )
                    else:
                        drone_df.to_excel(writer, sheet_name=drone_name, index=False)
                #pd.DataFrame(data_map).to_excel(writer, sheet_name="map", index=False)

    else:
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            for drone_name, drone_df in env.uav_data.items():
                drone_df.to_excel(writer, sheet_name=drone_name, index=False)
            #pd.DataFrame(data_map).to_excel(writer, sheet_name="map", index=False)
