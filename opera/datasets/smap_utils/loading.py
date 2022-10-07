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
        Return:
            results (dict): JointDatase 传递的字典。
            ['ann_info', 'img_prefix', 'seg_prefix', 
            'proposal_file', 'bbox_fields', 'mask_fields', 
            'keypoint_fields', 'filename', 'ori_filename', 'img', 
            'img_shape', 'ori_shape', 'img_fields']
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        if results['img_prefix'] is not None:
            # 注意这里对应的img_prefix是不同数据集自己的路径。
            filepath = osp.join(results['img_prefix'], 
                                results['ann_info']['img_paths'])
        else:
            raise ValueError("img_prefix 不能为空。")
        
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        
        if self.to_float32:
            img = img.astype(np.float32)
        
        results['filename'] = filepath
        results['ori_filename'] = results['ann_info']['img_paths']
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
class LoadAnnosFromFile(MMDetLoadAnnotations):
    """对注释文件进行进一步解析,转换格式，添加字段。
    Args:
        with_dataset:(bool):
        with_bbox:(bool):
    Return:
        add new keys:
            ['gt_bboxes', 'bbox_fields', 'dataset', 'gt_labels',
                'gt_keypoints', 'keypoints_fields', ['gt_vis_flag'], 'gt_areas']
    """
    def __init__(self, 
                *args,
                with_dataset=True, 
                with_keypoints=True,
                with_area=True,
                **kwargs):
        super(LoadAnnosFromFile, self).__init__(*args, **kwargs)
        self.with_dataset = with_dataset
        self.with_keypoints = with_keypoints
        self.with_area = with_area
    
    def _load_bboxes(self, results):
        bboxs = results['ann_info']['bboxs'].copy()
        bboxs = np.array(bboxs, dtype=np.float32)
        assert bboxs.shape[-1] == 4, f"the shape of bboxs is {bboxs.shape}"
        # 修改bboxs，coco中的格式为：[x1, y1, w, h]
        # 注意这里仅修改了results['bboxs'], [num_gts, 4]
        bboxs[:, 2] += bboxs[:, 0]
        bboxs[:, 3] += bboxs[:, 1]
        # 确保bbox中的坐标为[left_top_x, left_top_y, right_bottom_x, right_bottom_y]
        for i in range(len(bboxs)):
            left_top_x = min(bboxs[i][0], bboxs[i][2])
            left_top_y = min(bboxs[i][1], bboxs[i][3])
            right_bottom_x = max(bboxs[i][0], bboxs[i][2])
            right_bottom_y = max(bboxs[i][1], bboxs[i][3])
            bbox = [left_top_x, left_top_y, right_bottom_x, right_bottom_y]
            bboxs[i] = bbox
        results['gt_bboxes'] = bboxs
        results['bbox_fields'] = ['gt_bboxes']
        return results
    
    def _load_dataset(self, results):
        dataset = results['ann_info']['dataset'].upper()
        assert dataset in ['COCO', 'MUCO'], f"the dataset is {dataset}"
        results['dataset'] = dataset
        return results
    
    def _load_keypoints(self, results):
        """加载关键点的同时添加关键点标志位
        确保原来无效的坐标点，经过数据增强后依旧是(0, 0, ...)
        """
        # 加载关键点数据
        keypoints = results['ann_info']['bodys'].copy()  # [N, J, 11]
        keypoints = np.array(keypoints, dtype=np.float32)
        assert keypoints.shape[-2:] == (15, 11), \
        f"the shape of keypoints is {keypoints.shape}"
        results['gt_keypoints'] = keypoints
        results['keypoint_fields'] = ['gt_keypoints']
        return results
    
    def _load_areas(self, results):
        """加载areas
        用于计算oks loss
        """
        if results['dataset'] == "COCO":
            areas = results['ann_info']['areas'].copy()
        elif results['dataset'] == "MUCO":
            areas = []
            bboxs = results['gt_bboxes']
            for i in range(len(bboxs)):
                [left_top_x, left_top_y, right_bottom_x, right_bottom_y] = bboxs[i]
                area = (right_bottom_x - left_top_x) * (right_bottom_y - left_top_y)
                areas.append(area)
        results['gt_areas'] = np.array(areas, dtype=np.float32)
        results['areas_fields'] = ['gt_areas']
        return results
    
    def _load_labels(self, results):
        """原始annotations中不包括labels。
        由于label只有一种, 故根据bbox个数添加
        Args:
            results (_type_): _description_
        """
        bboxs = results['gt_bboxes']
        keypoints = results['gt_keypoints']
        assert bboxs.shape[0] == keypoints.shape[0], f"bboxs 和 keypoints的长度应该保持一致"
        num_person = bboxs.shape[0]
        labels = [0 for _ in range(num_person)]
        results['gt_labels'] = np.array(labels)
        return results

    def __call__(self, results):

        if self.with_dataset:
            results = self._load_dataset(results)
            
        if self.with_keypoints:
            results = self._load_keypoints(results)
        
        if self.with_bbox:
            results = self._load_bboxes(results)

        if self.with_area:
            results = self._load_areas(results)

        if self.with_label:
            results = self._load_labels(results)
        
        if results is None:
            return None

        return results
    
    def __repr__(self):
        repr_str = super(LoadAnnosFromFile, self).__repr__()[:-1] + ', '
        repr_str += f'with_dataset={self.with_dataset}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f'with_areas={self.with_areas}, '
        return repr_str
