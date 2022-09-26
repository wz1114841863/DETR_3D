# Document function description
#   测试dataset，data augment
from mmcv import Config, DictAction
from opera import __version__
from opera.apis import init_random_seed, set_random_seed, train_model
from opera.datasets import (build_dataset, build_dataloader)
from opera.models import build_model
import opera.datasets
import cv2 as cv
import torch


def main():
    cfg_file_path = "/data/jupyter/PETR/configs/_base_/datasets/coco_muco_keypoint_3d.py"
    cfg = Config.fromfile(cfg_file_path)
    dataset = [build_dataset(cfg.data.train)]
    for i in range(1020, 2000):
        data = dataset[0].__getitem__(i)
        import pdb;pdb.set_trace()
    # data = dataset[0].__getitem__(1023)
    # import pdb;pdb.set_trace()
    
if __name__ == '__main__':
    # print(opera.datasets.PIPELINES)
    main()