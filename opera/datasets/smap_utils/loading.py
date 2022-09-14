# 读取数据
import os.path as osp
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadAnnotations as MMDetLoadAnnotations

from ..builder import PIPELINES

@PIPELINES.register_module()
class LoadImgFromFile:
    """load an image form file
        参考mmdet.datasets.pipelines.loading.py
    """
    def __init__(self,
                    to_float32=False,
                    color_type='color',
                    channel_order='bgr',
                    file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None
    
    def __call__(self, results):
        """这里的results满足SMAP定义的格式
        存在两种情况：
            COCO:
            MuCo:
        Args:
            results (dict): JointDatase 传递的字典。
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        if results['img_prefix'] is not None:
            # 注意这里对应的img_prefix是不同数据集自己的路径。
            filepath = osp.join(results['img_prefix'], 
                                results['ann_info']['img_paths'])
        else:
            raise ValueError("img_prefix 不能为空。")
        
        img_bytes = self.file_client.get(filepath=filepath)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        
        if self.to_float32:
            img = img.astype(np.float32)
        
        results['filename'] = filepath
        results['ori_filename'] = results['ann_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results
    
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f"color_type='{self.color_type}', "
                f"channel_order='{self.channel_order}', "
                f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(MMDetLoadAnnotations):
    """对注释文件进行进一步解析
    
    Args:
        with_dataset:(bool):
        with_bbox:(bool):
    """
    def __init__(self, with_dataset=True, with_keypoints=True,
                    with_bbox=True, with_label=False, with_mask=False, 
                    with_seg=False, poly2mask=False, denorm_bbox=False, file_client_args=dict(backend='disk')):
        super().__init__(with_bbox, with_label, with_mask, with_seg, poly2mask, 
                            denorm_bbox, file_client_args)
        self.with_dataset = with_dataset
        self.with_keypoints = with_keypoints
    
    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['bboxs'] = ann_info['bboxs'].copy()
        # 修改bbox，coco中的格式为：[x1, y1, w, h]
        # 注意这里仅修改了results['bboxs']
        results['bboxs'][:, 2] += results['bboxs'][:, 0]
        results['bboxs'][:, 3] += results['bboxs'][:, 1]
        results['bbox_fields'] = ['bboxs']
        return results
    
    def _load_dataset(self, results):
        results['dataset'] = results['ann_info']['dataset'].copy()
        return results
    
    def _load_keypoints(self, results):
        results['keypoints'] = results['ann_info']['bodys'].copy()
        results['keypoints_fields'] = ['keypoints']
        return results
    
    def __call__(self, results):
        results = super(LoadAnnotations, self).__call__(results)
        
        if results is None:
            return None
        
        if self.with_dataset:
            results = self._load_dataset(results)
            
        if self.with_keypoints:
            results = self._load_keypoints(results)
            
        return results
    
    def __repr__(self):
        repr_str = super(LoadAnnotations, self).__repr__()[:-1] + ', '
        repr_str += f'with_keypoint={self.with_dataset}, '
        return repr_str
