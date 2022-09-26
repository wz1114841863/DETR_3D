# Document function description
#   用于加载于Normalize之前，保存图片并绘制bbox和keypoint。
import cv2 as cv
import os.path as osp
from ..builder import PIPELINES

@PIPELINES.register_module()
class VisImg:
    """从result中读取img, bbox, keypoints等数据
    绘制图片并保存。
    Args:
        draw_bbo: 是否绘制bbox，默认为True。
        draw_keypoints: 是否绘制keypoints，默认为True。
        img_prefix: 图片名称前缀，可用于描述图片是在什么阶段保存的。默认为空。
        img_path: 图片保存路径。默认为 './'
    """
    def __init__(self,
                    draw_bbox=True,
                    draw_keypoints=True,
                    img_prefix='',
                    img_path='./'):
        self.draw_bbox = draw_bbox
        self.draw_keypoints = draw_keypoints
        self.img_prefix = img_prefix
        self.img_path = img_path
        
    @staticmethod
    def _get_img(results):
        img = results['img'].copy()
        return img
    
    @staticmethod
    def _draw_bbox(img, results):
        bboxs = results['gt_bboxes']
        for bbox in bboxs:
            ptLeftTop = (int(bbox[0]), int(bbox[1]))
            ptRightBottom = (int(bbox[2]), int(bbox[3]))
            img = cv.rectangle(img, ptLeftTop, ptRightBottom, (0, 0, 255))
        return img
    
    @staticmethod
    def _draw_keypoints(img, results):
        persons_kps = results['gt_keypoints'].astype(int)
        point_size = 1
        point_color = (0, 0, 255)  # BGR
        thickness = 2
        for single_kps in persons_kps:
            for keypoint in single_kps:
                keypoint = keypoint.astype(int)
                if keypoint[3] == 0:
                    continue
                img = cv.circle(img, keypoint[:2], point_size, point_color, thickness)
        return img
        
    def _save_img(self, img):
        img_name = self.img_prefix + "_img.jpg"
        img_path = osp.join(self.img_path, img_name)
        cv.imwrite(img_path, img)
    
    def __call__(self, results):
        """绘制并保存图片

        Args:
            results (dict): 
                "gt_bboxs": [person_num, 4]
                "gt_keypoints": [person_num, J, 11]
                "img": img
                others: unused.
        """
        img = self._get_img(results)
        
        if self.draw_bbox:
            img = self._draw_bbox(img, results)
        
        if self.draw_keypoints:
            img = self._draw_keypoints(img, results)
        
        self._save_img(img)
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(draw_bbox={self.draw_bbox}, '
        repr_str += f'draw_keypoints={self.draw_keypoints}'
        return repr_str
