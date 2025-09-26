import argparse
import math
import re
from glob import glob
import csv
from collections import OrderedDict
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from data_loader import PUNET_Dataset
import os
import time
from pc_util import load
from data_util import normalize_point_cloud
from chamfer_distance.chamfer_distance import ChamferDistanceFunction as chamfer_3DDist
from train_option import get_train_options
from loss import Loss
parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=str, required=False, default="../outputs/exp", help=".xyz")  # 输出数据所在位置
#parser.add_argument("--gt", type=str, required=False, default="../Unsampling/gt", help=".xyz")
#parser.add_argument("--gt", type=str, required=False, default="../Unsampling/gt_4096", help=".xyz")
parser.add_argument("--gt", type=str, required=False, default="../Unsampling/gt_8192", help=".xyz")
FLAGS = parser.parse_args()
PRED_DIR = os.path.abspath(FLAGS.pred)  # 得到测试数据文件绝对路径
GT_DIR = os.path.abspath(FLAGS.gt)  # 拿到真实数据的绝对路径
chamfer_distance1 = chamfer_3DDist.apply




gt_paths = glob(os.path.join(GT_DIR, '*.xyz'))

pred_paths = glob(os.path.join(PRED_DIR, '*xyz'))

# print(gt_paths)
# print(pred_paths)
gt_names = [os.path.basename(p)[:-4] for p in gt_paths]  # os.path.basename(p)返回路径名中最后一个组成部分
pred_names = [os.path.basename(p)[:-4] for p in gt_paths]
# gt = gt_paths[0]
# gt = load(gt_paths[0])[:, :3]  # 只取第一个数据评估
# pred = load(pred_paths[0])[:, :3]
#
# pred_tensor, centroid, furthest_distance = normalize_point_cloud(gt)
# gt_tensor, centroid, furthest_distance = normalize_point_cloud(pred)
#
# cd_forward, cd_backward = chamfer_distance(torch.from_numpy(gt_tensor).cuda(), torch.from_numpy(pred_tensor).cuda())
# cd_forward = cd_forward[0, :].cpu().numpy()
# cd_backward = cd_backward[0, :].cpu().numpy()


def nearest_distance(queries, pc, k=2):
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    dis, knn_idx = knn_search.kneighbors(queries, return_distance=True)
    return dis[:, 1]


# 均匀度分析
def analyse_uniform(idx_file, radius_file, map_points_file):
    start_time = time()
    points = load(map_points_file)  # 球查询，以每个种子点为中心，r为半径的查询子集
    radius = np.loadtxt(radius_file)
    with open(idx_file) as f:
        lines = f.readlines()  # 读取txt文件的每一行

    sample_number = 1000
    rad_number = radius.shape[0]

    uniform_measure = np.zeros([rad_number, 1])
    densitys = np.zeros([rad_number, sample_number])

    expect_number = precentages * points.shape[0]

    expect_number = np.reshape(expect_number, [rad_number, 1])

    for j in range(rad_number):
        uniform_dis = []

        for i in range(sample_number):

            density, idx = lines[i * rad_number + j]
            densitys[j, i] = int(density)
            coverage = np.square(densitys[j, i] - expect_number[j]) / expect_number[j]

            num_points = re.findall("(\d+)", idx)

            idx = list(map(int, num_points))
            if len(idx) < 5:
                continue
            idx = np.array(idx).astype(np.int32)
            map_point = points[idx]

            shortest_dis = nearest_distance(map_point, map_point, k=2)
            disk_area = math.pi * (radius[j] ** 2) / map_point.shape[0]
            expect_d = math.sqrt(2 * disk_area / 1.732)  # using hexagon 使用六边形

            dis = np.square(shortest_dis - expect_d) / expect_d
            dis_mean = np.mean(dis)  # 采用一个平均值
            uniform_dis.append(coverage * dis_mean)

        uniform_dis = np.array(uniform_dis).astype(np.float32)
        uniform_measure[j, 0] = np.mean(uniform_dis)
    return uniform_measure


precentages = np.array([0.004, 0.006, 0.008, 0.01, 0.012])
fieldnames = ["CD", "hausdorff", "p2f avg", "p2f", "p2f std"]
fieldnames += ["uniform_%d" % d for d in range(precentages.shape[0])]
for D in [PRED_DIR]:
    avg_md_forward_value = 0
    avg_md_backward_value = 0
    avg_hd_value = 0
    counter = 0
    pred_paths = glob(os.path.join(D, "*.xyz"))

    gt_pred_pairs = []
    for p in pred_paths:
        name, ext = os.path.splitext(os.path.basename(p))
        assert (ext in (".ply", ".xyz"))
        try:
            gt = gt_paths[gt_names.index(name)]
        except ValueError:
            pass
        else:
            gt_pred_pairs.append((gt, p))


    tag = re.search("/(\w+)/result", os.path.dirname(gt_pred_pairs[0][1]))
    if tag:
        tag = tag.group()[0]
    else:
        tag = D

    global_p2f = []
    global_density = []
    global_uniform = []

    with open(os.path.join(os.path.dirname(gt_pred_pairs[0][1]), "evaluation.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()  # 只使用预先指定的字段名写入csv文件的第一行
        cd_forward_value_list=[]
        cd_backward_value_list=[]
        hd_value_list =[]
        row = {}
        for gt_path, pred_path in gt_pred_pairs:
            gt = load(gt_path)[:, :3]
            gt = gt[np.newaxis, ...]  # np.newaxis增加一个维度，行增加一个维度 ,为了求CD距离
            pred = load(pred_path)
            pred = pred[:, :3]

            pred = pred[np.newaxis, ...]  #1 ,1024,3
            pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred)
            gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt)
            cd_forward, cd_backward = chamfer_distance1(torch.from_numpy(gt_tensor).cuda(),
                                                       torch.from_numpy(pred_tensor).cuda())
            cd_forward_value=cd_forward[0, :].cpu().numpy()
            cd_backward_value =cd_backward[0, :].cpu().numpy()
            #print(cd_forward)
           # print(cd_forward_value)
           # cd_forward_value, cd_backward_value = [cd_forward, cd_backward]
           # print("=======================================")
            md_value = np.mean(cd_forward_value) + np.mean(cd_backward_value)
            hd_value = np.max(np.amax(cd_forward_value, axis=0) + np.amax(cd_backward_value, axis=0))
            cd_forward_value = np.mean(cd_forward_value)
            cd_backward_value = np.mean(cd_backward_value)
            row["CD"] = cd_forward_value + cd_backward_value
            row["hausdorff"] = hd_value
            cd_backward_value_list.append(cd_backward_value)
            cd_forward_value_list.append(cd_forward_value)
           # print(cd_forward_value)
           # print(cd_backward_value)
           #  avg_md_forward_value += cd_forward_value
           #  print("==================================")
           #  print(avg_md_backward_value)
           #  avg_md_backward_value += cd_backward_value
           # avg_hd_value += hd_value
            hd_value_list.append(hd_value)
            if os.path.isfile(pred_path[: -4] + "_point2mesh_distance.txt"):
                point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.txt")
                if point2mesh_distance.size == 0:
                    continue
                point2mesh_distance = point2mesh_distance[:, 3]
                global_p2f.append(point2mesh_distance)

            if os.path.isfile(pred_path[:-4] + "_disk_idx.txt"):
                idx_file = pred_path[:-4] + "_disk_idx.txt"
                radius_file = pred_path[:-4] + '_radius.txt'
                map_point_file = pred_path[:-4] + '_point2mesh_distance.txt'

            counter += 1
        # print(cd_backward_value_list)
        # print(cd_forward_value_list)
        # md_value = np.mean(cd_forward_value_list) + np.mean(cd_backward_value_list)
        # hd_value = np.max(np.amax(cd_forward_value_list, axis=0) + np.amax(cd_backward_value_list, axis=0))
        # cd_forward_value = np.mean(cd_forward_value_list)
        # cd_backward_value = np.mean(cd_backward_value_list)
        # avg_md_forward_value += cd_forward_value
        # avg_md_backward_value += cd_backward_value
        # avg_hd_value += hd_value
        row = OrderedDict()
        # print(avg_md_forward_value)
        # print(avg_md_backward_value)
        # print(counter)
        # avg_md_forward_value /= counter
        # avg_md_backward_value /= counter
        # avg_hd_value /= counter
        # avg_cd_value = avg_md_forward_value + avg_md_backward_value
        avg_cd_value =np.sum(cd_forward_value_list+cd_backward_value_list)/counter
        avg_hd_value =np.sum(hd_value_list)/counter
        print(avg_cd_value*1000)
        print(avg_hd_value*1000)
        row["CD"] = avg_cd_value*1000
        row["hausdorff"] = avg_hd_value*1000
        if global_p2f:
            global_p2f = np.concatenate(global_p2f, axis=0)
            mean_p2f = np.nanmean(global_p2f)
            row["p2f avg"] = mean_p2f

        if global_uniform:
            global_uniform = np.array(global_uniform)
            uniform_mean = np.mean(global_uniform, axis=0)
            for i in range(precentages.shape[0]):
                row["uniform_%d" % i] = uniform_mean[i, 0]

        writer.writerow(row)
print("scessful")

