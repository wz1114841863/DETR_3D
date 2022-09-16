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
    cfg_file_path = "/data/jupyter/PETR/configs/_base_/datasets/smap_keypoint.py"
    cfg = Config.fromfile(cfg_file_path)
    dataset = [build_dataset(cfg.data.train)]
    data = dataset[0].__getitem__(10)
    # train_dataloader_default_args = dict(
    #     samples_per_gpu=1,
    #     workers_per_gpu=1,
    #     # `num_gpus` will be ignored if distributed
    #     num_gpus=1,
    #     dist=False,
    #     seed=8888,
    #     runner_type='EpochBasedRunner',
    #     persistent_workers=False
    # )
    # train_loader_cfg = {
    #     **train_dataloader_default_args,
    #     **cfg.data.get('train_dataloader', {})
    # }
    # data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    # data_loader = data_loaders[0]
    # for i, data_batch in enumerate(data_loader):
    #     print(len(data_batch))
    
if __name__ == '__main__':
    # print(opera.datasets.PIPELINES)
    main()