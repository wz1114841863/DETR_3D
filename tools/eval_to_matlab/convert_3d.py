import numpy as np
import scipy.io as scio
import json

def convert(path=''):
    with open(path, 'r') as f:
        data = json.load(f)

    pairs_3d = data['3d_pairs']

    pose3d = dict()
    pose2d = dict()
    gt3d = dict()
    # count = dict()

    for i in range(len(pairs_3d)):  
        name = pairs_3d[i]['image_path']
        name = name[name.index('TS'):]
        pred_3ds = np.array(pairs_3d[i]['pred_3d'])  # [num_gt, 15, 4]
        gt_3ds = np.array(pairs_3d[i]['gt_3d'])  # [num_gts, 16, 7]
        pred_2ds = pairs_3d[i]['pred_2d'] 
        pred_2ds = np.array(pred_2ds)  # [num_gts, 15, 4]
        pose3d[name] = pred_3ds * 10 # nH x 15 x 4 
        pose3d[name][:, :, 3] /= 10
        pose2d[name] = pred_2ds # nH x 15 x 4
        gt3d[name] = gt_3ds * 10     # nH x 15 x 4        

    if True:
        scio.savemat('./pose3d.mat', {'preds_3d_kpt':pose3d})
        scio.savemat('./pose2d.mat', {'preds_2d_kpt':pose2d})

if __name__ == "__main__":
    file_path = "/home/notebook/code/personal/S9043252/wz/PETR_3D/work_dirs/3d_0930_train_result/output.json"
    convert(file_path)