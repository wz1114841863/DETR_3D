# Document function description
#   模仿SMAP的数据格式，混合加载已经修改格式之后的COCO和MuCo数据集。
from torch.utils.data import Dataset
import mmcv
import numpy as np
import copy
import os.path as osp
import torch

from .builder import DATASETS


@DATASETS.register_module()
class JointDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, with_augmentation=False):
        """_summary_

        Args:
            cfg (_type_): _description_
            mode (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
            with_augnentation (bool, optional): _description_. Defaults to False.
        """
        assert mode in ('train', 'test', 'generation')
        self.mode = mode
        self.transform = transform
        self.train_data = list()
        self.val_data = list()
        
        DATASET = cfg.dataset
        if self.mode == "train":
            data = []
            # 使用COCO和MuCo做为混合数据集
            with open(DATASET.COCO_JSON_PATH) as fp:  # 这里的json是已经处理为SAMP注释格式的
                data = mmcv.load(fp)['root']

            # 加载使用的3D数据集，这里仅使用MuCo
            for dataset_name in DATASET.USED_3D_DATASETS: # 'MUCO', 'CMUP', 'H36M'
                with open(eval('DATASET.%s_JSON_PATH'%(dataset_name))) as fp:
                    data_3d = mmcv.load(fp)['root']
                    data += data_3d  
            del data_3d
        elif self.mode == "generation":
            data = []
            for dataset_name in DATASET.USED_3D_DATASETS: # 'MUCO', 'CMUP', 'H36M'
                with open(eval('DATASET.%s_JSON_PATH'%(dataset_name))) as fp:
                    data_3d = mmcv.load(fp)['root']
                    data += data_3d
            del data_3d
        else:
            with open(cfg.TESE.JSON_PATH) as fp:
                data = mmcv.load(fp)['root']
                
        for i in range(len(data)):
            if data[i]['isValidation'] != 0:
                self.val_data.append(data[i])
            else:
                self.train_data.append(data[i])  # 254564
        
        # keypoints information
        self.root_idx = DATASET.ROOT_IDX  # 2
        self.keypoint_num = DATASET.KEYPOINT.NUM  # 15
        
        # data root path
        self.test_root_path = cfg.TEST.ROOT_PATH
        self.root_path = {}
        for dname in (['COCO'] + DATASET.USED_3D_DATASETS): # 'MUCO', 'CMUP', 'H36M'
            self.root_path[dname] = eval('DATASET.%s_ROOT_PATH'%(dname)) 
    
        self.max_people = cfg.DATASET.MAX_PEOPLE  # 20

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        else:
            return len(self.val_data)
        
    def get_anno(sefl, data):
        anno = dict()
        anno['dataset'] = data['dataset'].upper()
        anno['img_height'] = int(data['img_height'])
        anno['img_width'] = int(data['img_width'])
        anno['isValidation'] = data['isValidation']
        anno['bodys'] = np.asarray(data['bodys'])
        anno['center'] = np.array([anno['img_width'] // 2, anno['img_height'] // 2])
        return anno
    
    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'generation':
            data = copy.deepcopy(self.train_data[index])
        else:
            data = copy.deepcopy(self.val_data[index])
            
        meta_data = self.get_anno(data)
        if self.mode not in ['train', 'generation']:
            root_path = self.test_root_path  # 自定义数据路径
        else:
            root_path = self.root_path[meta_data['dataset']]
            
        img = mmcv.imread(osp.join(root_path, data['img_paths']))
        if self.with_augmentation:
            pass  # 数据增强
        
        if self.transform:
            img = self.transform(img)
        else:
            img = img.transpose((2, 0, 1)).astype(np.float32)
            img = torch.from_numpy(img).float
        
        if self.mode in ['test', 'generation']:
            pass
        
        valid = np.ones((self.keypoint_num, 1), np.float)  # [15, 1]
        if  meta_data['dataset'] == 'COCO':
            pass
        
        return img, meta_data
        
if __name__ == '__main__':
    pass

