# Document function description
#   模仿SMAP的数据格式，加载已经修改格式之后的COCO和MuCo数据集。
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
        return self.data_infos[idx]
    
    def prepare_train_img(self, idx):
        """获取管道后的训练数据和注释文件
        Args:
            idx (int): 数据索引
        
        Returns:
            dict: 处理后的训练数据和注释文件。
        """
        
        ann_info = self.get_ann_info(idx=idx)
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
        results['seg_fields'] = []
        results['keypoint_fields'] = []
        results['area_fields'] = []
    
    def get_anno(sefl, data):
        anno = dict()
        anno['dataset'] = data['dataset'].upper()
        anno['img_height'] = int(data['img_height'])
        anno['img_width'] = int(data['img_width'])
        anno['isValidation'] = data['isValidation']
        anno['bodys'] = np.asarray(data['bodys'])
        anno['center'] = np.array([anno['img_width'] // 2, anno['img_height'] // 2])
        return anno
    
if __name__ == '__main__':
    pass

