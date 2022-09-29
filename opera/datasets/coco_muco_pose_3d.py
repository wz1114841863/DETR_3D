# Document function description
#   模仿SMAP的数据格式，加载已经修改格式之后的COCO和MuCo数据集。
import warnings
from mmdet.datasets import CustomDataset
import mmcv
import numpy as np
import json


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
    CLASSES = ('person', )
    
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
                [7, 8],]
    
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
            elif data['gt_bboxes']._data.shape[0] == 0:
                # print(f"长度为0")
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

    def evaluate(self,
                    results,
                    anno_path,
                    metric='keypoints',
                    logger=None,
                    jsonfile_prefix=None,
                    classwise=False,
                    proposal_nums=(100, 300, 1000),
                    iou_thrs=None,
                    metric_items=None):
        """用于生成输出的json文件。

        Args:
            results (_type_): 网络输出结果
            anno_path: MuPoTs.json文件存放位置
            metric (str, optional): _description_. Defaults to 'keypoints'.
            logger (_type_, optional): _description_. Defaults to None.
            jsonfile_prefix (_type_, optional): _description_. Defaults to None.
            classwise (bool, optional): _description_. Defaults to False.
            proposal_nums (tuple, optional): _description_. Defaults to (100, 300, 1000).
            iou_thrs (_type_, optional): _description_. Defaults to None.
            metric_items (_type_, optional): _description_. Defaults to None.
        """
        # 读取json文件
        with open(anno_path, 'w') as fp:
            annos = json.load(fp)['root']
        
        assert len(annos) == len(results), f"len(anno) != len(results), {len(anno)}"
        for i in range(len(annos)):
            anno = annos[i]
            result = results[i]
            # 取出result中对应数据
            bboxes, kpts,  depths= result[0][0], result[1][0], result[2][0]
            assert bboxes.shape == (100, 5), \
                f"error. bboxes.shape:{bboxes.shape}"
            assert kpts.shape == (100, 5), \
                f"error. kpts.shape:{bboxes.shape}"
            assert depths.shape == (100, 5), \
                f"error. depths.shape:{bboxes.shape}"            
            # 
        
        
    def __repr__(self):
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                    f'with number of images {len(self)}, \n')
        return result

if __name__ == '__main__':
    pass

