# Document function description
#   模仿SMAP的数据格式，加载已经修改格式之后的COCO和MuCo数据集。
import warnings
from mmdet.datasets import CustomDataset
import mmcv
import numpy as np


from .builder import DATASETS


@DATASETS.register_module()
class JointDataset(CustomDataset):
    """
    MuCo keypoint indexes::
        0 : "neck",
        1 : "head",
        2 : "pelvis",  # root
        3 : "LShoulder",
        4 : "LElbow",
        5 : "LWrist",
        6 : "LHip",
        7 : "LKnee",
        8 : "LAnkle",
        9 : "RShoulder",
        10: "RElbow",
        11: "RWrist",
        12:" RHip",
        13: "RKnee",
        14: "RAnkle "
    """
    CLASSES = ('person')
    
    PALETTE = [[0, 1], 
                [0, 2], 
                [0, 9], 
                [9, 10], 
                [10, 11], 
                [0, 3], 
                [3, 4], 
                [4, 5], 
                [2, 12], 
                [12, 13], 
                [13, 14], 
                [2, 6], 
                [6, 7], 
                [7, 8]]
    
    def __init__(self, 
                    *args,
                    **kwargs):
        super(JointDataset, self).__init__(*args, **kwargs)
    
    def load_annotations(self, ann_file):
        """加载注释文件, 注意这里使用的注释文件为SMAP提出的注释格式
        Args:
            ann_file(str): 文件路径 
        Returns:
            annos(dict): 注释文件内容
        """
        return mmcv.load(ann_file)['root']
    
    def get_ann_info(self, idx):
        """获取注释文件信息
        Args:
            idx (int): 数据索引
        Returns:
            _type_: _description_
        """
        
        return self.data_infos[idx]
    
    def get_cat_ids(self, idx):
        """根据索引获取类别
        Args:
            idx (int): 数据索引
        Returns:
            list[int]: 图片中包含的class
        """
        return self.data_infos[idx]['cam_id']
    
    def prepare_train_img(self, idx):
        """获取管道后的训练数据和注释文件
        Args:
            idx (int): 数据索引
        anno_info :['dataset', 'img_paths', 'img_width', 'img_height', 
            'image_id', 'cam_id', 'bodys', 'bboxs', 'num_keypoints', 
            'iscrowd', 'segmentation', 'isValidation'])
        Returns:
            dict: 处理后的训练数据和注释文件。
        """
        ann_info = self.get_ann_info(idx) 
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results=results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['keypoint_fields'] = []
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict:
                "meta_data":
                    'filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg'
                "img":
                "gt_bbox":
                "gt_keypoints":
                "ann_info":
                "dataset":
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
    def _filter_imgs(self, min_size=32):
        # if self.filter_empty_gt:
        #     warnings.warn(
        #         'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['img_width'], img_info['img_height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['img_width'] / img_info['img_height'] > 1:
                self.flag[i] = 1
    
    def __repr__(self):
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                    f'with number of images {len(self)}, \n')
        return result
if __name__ == '__main__':
    pass

