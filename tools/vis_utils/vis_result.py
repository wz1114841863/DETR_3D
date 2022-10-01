# 获取数据集图片和网络输出，进行可视化
import cv2 as cv
import numpy as np
import json


json_path = "/home/notebook/code/personal/S9043252/wz/PETR_3D/work_dirs/3d_0930_train_result/output.json"
dataset_path = "/home/notebook/data/group/wangxiong/smoothformer/hpe_data/data/MuPoTs/MultiPersonTestSet/"
keys = ['gt_2d', 'gt_3d', 'pred_2d', 'pred_3d', 'root_d']


with open(json_path, 'r') as fp:
    anno = json.load(fp)['3d_pairs']

for i in range(2000, len(anno)):
    img_path = dataset_path + anno[i]['image_path']
    img = cv.imread(img_path)
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    gt_point_color = (255, 0, 0)  # BGR
    thickness = 2
    gt_2ds = np.asarray(anno[i]['gt_2d'])
    pred_2ds = np.asarray(anno[i]['pred_2d'])

    for j in range(len(gt_2ds)):
        for k in range(16):
            gt_keypoint = gt_2ds[j][k][:2].astype(int)
            img = cv.circle(img, gt_keypoint, point_size, gt_point_color, thickness)

    for j in range(len(pred_2ds)):
        for k in range(15):
            keypoint = pred_2ds[j][k][:2].astype(int)
            img = cv.circle(img, keypoint, point_size, point_color, thickness)

    tmp_save_path = "/home/notebook/code/personal/S9043252/wz/PETR_3D/tools/vis_utils/tmp.jpg"
    img = cv.imwrite(tmp_save_path, img)
    
    import pdb;pdb.set_trace()