#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
"""
File:intersect

"""
import math

import numpy as np


# 求向量ab和向量cd的叉乘
def xmult(a, b, c, d):
    vectorAx = b[0] - a[0]
    vectorAy = b[1] - a[1]
    vectorBx = d[0] - c[0]
    vectorBy = d[1] - c[1]
    return vectorAx * vectorBy - vectorAy * vectorBx


# 判断线段ab和线段cd是否相交,相交返回True，不相交返回False
def cross(a, b, c, d):  # start1,end1,start2,end2
    # 以c为公共点，分别判断向量cd到向量ca与到向量cb的方向，记为xmult1和xmult2。
    # 若ab分布于cd两侧，xmult1 * xmult2应小于0。
    # 同理若cd分布于ab两侧，xmult3 * xmult4应小于0。
    xmult1 = xmult(c, d, c, a)
    xmult2 = xmult(c, d, c, b)
    xmult3 = xmult(a, b, a, c)
    xmult4 = xmult(a, b, a, d)
    if xmult1 * xmult2 < 0 and xmult3 * xmult4 < 0:
        return True
    else:
        return False


def cell_intersect(startl, endl, rec):  # 线段l，矩形rec
    # print(rec)
    if rec[0] < startl[0] < rec[0] + 1 and rec[1] < startl[1] < rec[1] + 1:
        # print("线段起点在格子内")
        return True
    elif rec[0] < endl[0] < rec[0] + 1 and rec[1] < endl[1] < rec[1] + 1:
        # print("线段终点在格子内")
        return True
    else:
        # print("线段两端点不在格子内")
        # 判断是否与对角线相交
        diag1s = rec
        diag1e = (rec[0] + 1, rec[1] + 1)
        diag2s = (rec[0], rec[1] + 1)
        diag2e = (rec[0] + 1, rec[1])
        if cross(diag1s, diag1e, startl, endl):
            # print("线段与对角线1相交")
            return True
        elif cross(diag2s, diag2e, startl, endl):
            # print("线段与对角线2相交")
            return True
        else:
            return False


def ac_detect(pos, action):
    start = np.array(pos[:2])
    end_x, end_y = round(start[0] + math.cos(action[0]) * action[1]), round(
        start[1] + math.sin(action[0]) * action[1]
    )
    scell = [start[0], start[1]]
    start = np.array([pos[0] + 0.5, pos[1] + 0.5])
    end = np.array([end_x + 0.5, end_y + 0.5])

    ecell = [end_x, end_y]
    # new_poses = np.array([scell, ecell])
    new_poses = np.empty([0, 2])
    a_x = 1
    a_y = 1

    if scell[0] > ecell[0]:
        a_x = -1
    if scell[1] > ecell[1]:
        a_y = -1

    for i in range(scell[0], ecell[0] + a_x, a_x):
        for j in range(scell[1], ecell[1] + a_y, a_y):
            cell = np.array([i, j])
            if cell_intersect(start, end, cell):
                cell = np.array([i, j]).reshape(1, 2)
                new_poses = np.append(new_poses, values=cell, axis=0)
    return new_poses


def dist(a, b):
    d = math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))
    return d


def collision_detect(a, b, c, d, ls):  # uav1:a--b uav2:c--d ls安全距离
    print("\n检测无人机动作")
    if dist(a, c) < ls:
        print("两无人机起点距离小于安全距离")
        return False
    elif dist(b, d) < ls:
        print("两无人机终点距离小于安全距离")
        return False
    else:
        # 判断是否中途发生碰撞
        if cross(a, b, c, d):
            print("两无人机中途发生碰撞")
            return False
        else:
            print("两无人机安全飞行")
            return True
