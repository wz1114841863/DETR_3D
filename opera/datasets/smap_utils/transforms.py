import random
import numpy as np
import cv2 as cv
import mmcv
from mmdet.datasets.pipelines import RandomFlip as MMDetRandomFlip
from mmdet.datasets.pipelines import Resize as MMDetResize
from mmdet.datasets.pipelines import RandomCrop as MMDetRandomCrop

from ..builder import PIPELINES


FLIP_ORDER = [0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]

"""
以下的操作均不涉及3D坐标, 可能存在一定的问题，待解决。
目前仅对 [x, y, Z, v, X, Y, Z, fx, fy, cx, cy] 中
    x, y,做相应的变换处理。
还需要对 X, Y, Z, fx, fy, cx, cy 等做处理。
"""


@PIPELINES.register_module()
class AugRandomFlip(MMDetRandomFlip):
    """随机翻转
    分为两种情况：
        COCO:
        MuCo:
    Return:
        add new keys:
            ['flip', 'flip_direction']
    """
    def __init__(self, flip_ratio=0.8, direction='horizontal'):
        assert direction == 'horizontal', \
            "仅支持水平翻转，不支持其余方向"
        super().__init__(flip_ratio, direction)
    
    def keypoint_flip(self, results):
        """水平翻转关键点坐标。

        注: 这里仅翻转了2d坐标中的x和交换了对应的关键点坐标, 但并未修改3d 坐标中的
        X, 是否存在问题。
        """
        # 翻转所有关键点
        keypoints = results['gt_keypoints'].copy()
        gt_keypoints_flag = results['gt_keypoints_flag'].copy()
        for i in range(len(keypoints)):  # [person_num, 15, 11]
            # change the coordinate
            keypoints[i][:, 0] = results['img_shape'][1] - 1 - keypoints[i][:, 0]  # 根据图片中心翻转x
            # change the left and the right
            keypoints[i][:, :] = keypoints[i][FLIP_ORDER, :]
            gt_keypoints_flag[i][:, :] = gt_keypoints_flag[i][FLIP_ORDER, :]
        
        # 无效关键点重新置0
        for n in range(len(keypoints)):
            for j in range(15):
                if gt_keypoints_flag[n][j][0] == 0:
                    keypoints[n][j] = [0 for _ in range(11)]
        
        results['gt_keypoints_flag'] = gt_keypoints_flag
        results['gt_keypoints'] = keypoints
        return results  
    
    def img_flip(self, results):
        """翻转图片：

        Args:
            results (_type_): _description_
        """
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imflip(
            results[key], direction=results['flip_direction'])

        return results
        
    def bbox_flip(self, results):
        """对bbox进行翻转。

        Args:
            results:
        """
        img_shape = results['img_shape']  # (h, w)
        for key in results.get('bbox_fields', ['gt_bboxs']):
            bboxs = results[key]  # [person_num, 4]")
            assert bboxs.shape[1] == 4, "the shape of bbox isn't 4."
            flip_bboxs = bboxs.copy()
            flip_bboxs[:, 0] = img_shape[1] - 1 - bboxs[:, 0]
            flip_bboxs[:, 2] = img_shape[1] - 1 - bboxs[:, 2]
            flip_bboxs[:,[0, 2]] = flip_bboxs[:,[2, 0]]
            results[key] = flip_bboxs
            
        return results
    
    def __call__(self, results):
        """翻转img,bbox和keypoints。

        Args:
            results (dict): 传递过来的字典。
        """
        if 'flip' not in results:
            if isinstance(self.direction, list):
                direction_list = self.direction + [None]
            else:
                direction_list = [self.direction, None]
            
            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [non_flip_ratio]
            
            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            results['flip'] = cur_dir is not None
        
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        
        if results['flip']:  #如果为True，才进行翻转
            # flip image
            results = self.img_flip(results)
            # flip bbox
            results = self.bbox_flip(results)
            # flip keypoints
            results = self.keypoint_flip(results)

        return results


@PIPELINES.register_module()
class AugRandomRotate():
    """旋转图片,获取旋转矩阵
    
    Return:
        add new keys:

    """
    def __init__(self, 
                max_rotate_degree=0,
                rotate_prob=0.5,
                bordervalue=(128, 128, 128)):
        self.max_rotate_degree = max_rotate_degree
        self.bordervalue = bordervalue
        self.rotate_prob = rotate_prob
        if random.random() < self.rotate_prob:
            self.rotate = True
        else:
            self.rotate = False
    
    def _rotate_bound(self, results):
        """The correct way to rotation an image

        """
        for key in results.get('img_fields', ['img']):
            # grab the dimensions of the image and then determine the center
            (h, w) = results[key].shape[:2]
            (cX, cY) = (w // 2, h // 2)
            
            # 获取旋转矩阵
            angle = (random.random() - 0.5) * 2 * self.max_rotate_degree
            M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)  # M.shape: (2, 3)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            
            # 重新计算边界
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            
            # 执行实际旋转并返回图像
            img_rot = cv.warpAffine(results[key], M, (nW, nH), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT,
                            borderValue=self.bordervalue)
            
            # 注：这里图片的尺寸是如何变换的，是否需要更新参数。
            results[key] = img_rot  # (nH, nW, 3)
        
        results['img_shape'] = results['img'].shape
        self.M = M
        
        return results
    
    def _rotate_coordinate(self,p2d, M):
        aug_p2d = np.concatenate((p2d, np.ones((p2d.shape[0], 1))), axis=1)
        rot_p2d = (M @ aug_p2d.T).T
        return rot_p2d[:, :2]

    def _rotate_skel2d(self, results):
        """旋转2d坐标

        """
        # 旋转所有2d坐标
        for key in results.get('keypoint_fields', ['gt_keypoints']):
            keypoints = results[key].copy()
            for i in range(len(keypoints)):  # [person_num, 15, 11]
                keypoints[i][:, :2] = self._rotate_coordinate(keypoints[i][:, :2], self.M)
            
            
            # 无效关键点重新置0
            for n in range(len(keypoints)):
                for j in range(15):
                    if results['gt_keypoints_flag'][n][j] == 0:
                        keypoints[n][j] = [0 for _ in range(11)]
                        
            results[key] = keypoints

        return results
    
    def _rotate_bbox(self, results):
        """旋转bbox

        Args:
            results (_type_): _description_
        """
        for key in results.get('bbox_fields', ['gt_bboxs']):
            bboxs = results[key].copy()
            for i in range(len(bboxs)):  # [person_num, 4]
                bbox = bboxs[i]
                
                top_left = np.asarray((min(bbox[0], bbox[2]), min(bbox[1], bbox[3])))
                top_right = np.asarray((max(bbox[0], bbox[2]), min(bbox[1], bbox[3])))
                bottom_left = np.asarray((min(bbox[0], bbox[2]), max(bbox[1], bbox[3])))
                bottom_right = np.asarray((max(bbox[0], bbox[2]), max(bbox[1], bbox[3])))
                
                assert len(top_left) == 2 and len(top_right) == 2 and \
                    len(bottom_left) == 2 and len(bottom_right) == 2
                    
                coords = np.stack([top_left, top_right, bottom_left, 
                                    bottom_right], axis=0)
                assert coords.shape == (4, 2), \
                    f"error. the shape of coords is: {coords.shape}"
                coords = self._rotate_coordinate(coords, self.M)
                
                # 重新框定bbox: x1, y1, x2, y2
                [min_x, min_y] = np.min(coords, axis=0)
                [max_x, max_y] = np.max(coords, axis=0)
                # 注: 如果图片进行填充，这里的bbox就不会越出边界。
                bboxs[i] = [min_x, min_y, max_x, max_y]
            # 越界处理
            img_shape = results['img_shape']
            bboxs[:, 0::2] = np.clip(bboxs[:, 0::2], 0, img_shape[1])
            bboxs[:, 1::2] = np.clip(bboxs[:, 1::2], 0, img_shape[0])  
            results[key] = bboxs
        return results
    
    def __call__(self, results):
        """对img, bbox, keypoint执行旋转操作

        """
        if self.rotate:
            results = self._rotate_bound(results)
            results = self._rotate_skel2d(results)
            results = self._rotate_bbox(results)
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'rotate_probe={self.rotate_prob}'
        return repr_str


@PIPELINES.register_module()
class AugResize(MMDetResize):
    """Resize img, bbox, keypoints
    
    Return:
        add new keys:
            ['scale', 'scale_idx', 'scale_factor', 'keep_ratio']
    """
    def __init__(self, 
                    *args,
                    keypoint_clip_border=True,
                    **kwargs):
        super(AugResize, self).__init__(*args, **kwargs)
        self.keypoint_clip_border = keypoint_clip_border
    
    def _resize_img(self, results):
        """Resize img

        """
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:  # 保持纵横比
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
            
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
            results['img_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio
            
    def _resize_bboxes(self, results):
        return super()._resize_bboxes(results)
    
    def _resize_keypoints(self, results):
        # Resize 所有关键点
        for key in results.get('keypoint_fields', ['gt_keypoints']):
            keypoints = results[key].copy()
            keypoints[..., 0] = keypoints[..., 0] * results['scale_factor'][0]
            keypoints[..., 1] = keypoints[..., 1] * results['scale_factor'][1]
            
            # 关键点越界处理
            if self.keypoint_clip_border:
                img_shape = results['img_shape']
                keypoints[:, 0] = np.clip(keypoints[:, 0], 0, img_shape[1])
                keypoints[:, 1] = np.clip(keypoints[:, 1], 0, img_shape[0])
        
        # 无效关键点重新置0
        for n in range(len(keypoints)):
            for j in range(15):
                if results['gt_keypoints_flag'][n][j][0] == 0:
                    keypoints[n][j] = [0 for _ in range(11)]
                    
        results[key] = keypoints            

    def __call__(self, results):
        """进行一系列resize操作, img、bbox、keypoints

        Args:
            results (dict): _description_
        """
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)
        
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_keypoints(results)
        
        return results

    def __repr__(self):
        repr_str = super(AugResize, self).__repr__()[:-1] + ', '
        repr_str += f'keypoint_clip_border={self.keypoint_clip_border})'
        return repr_str


@PIPELINES.register_module()
class AugCrop(MMDetRandomCrop):
    """随机裁剪 img, bbox, keypoints

    """
    def __init__(self, 
                    *args,
                    kpt_clip_border=True,
                    **kwargs):
        super(AugCrop, self).__init__(*args, **kwargs)
        self.kpt_clip_border = kpt_clip_border
        
    def _crop_data(self, results, crop_size, allow_negative_crop):
        assert crop_size[0] > 0 and crop_size[1] > 1 , \
            "crop size 不符合范围"
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
            
            # crop image: [H, W, C]
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
            
        results['img_shape'] = img_shape        

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', ['gt_bboxs']):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                    dtype=np.float32)
            bboxes = results[key] - bbox_offset  # 注意这里是减去偏移。
            # 判断是否超出范围, 这里要确保bbox中的数据满足[x1, y1, x2, y2]
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            if (not valid_inds.any() and not allow_negative_crop):
                print(f"{results['img_prefix']} 裁剪后不包含有效的bbox")
                return None
            
            # 留取有效的bbox[N, 4] 和 keypoints[N, J, 11]
            results[key] = bboxes[valid_inds, :]
            results["gt_keypoints"] = results["gt_keypoints"][valid_inds]
            assert len(results[key]) == len(results["gt_keypoints"]), \
                "bbox长度与keypoints长度不一致"
            
            for key in results.get('keypoint_fields', ['gt_keypoints']):
                kpt_offset = np.array([offset_w, offset_h], dtype=np.float32)
                keypoints = results[key].copy()
                for single_person_keypoints in keypoints:
                    single_person_keypoints[:, :2] = \
                        single_person_keypoints[:, :2] - kpt_offset  # x, y - oofset_x, offset_y
                    invalid_idx = \
                        (single_person_keypoints[..., 0] < 0).astype(np.int8) | \
                        (single_person_keypoints[..., 1] < 0).astype(np.int8) | \
                        (single_person_keypoints[..., 0] > img_shape[1]).astype(np.int8) | \
                        (single_person_keypoints[..., 1] > img_shape[0]).astype(np.int8)
                    single_person_keypoints[invalid_idx > 0, :] = [0 for _ in range(11)]
                    
                    if self.kpt_clip_border:  # 可能并没有作用
                        single_person_keypoints[:, 0] = np.clip(
                            single_person_keypoints[..., 0], 0, img_shape[1])
                        single_person_keypoints[:, 1] = np.clip(
                            single_person_keypoints[..., 1], 0, img_shape[0])
                    
                # 无效关键点重新置0
                for n in range(len(keypoints)):
                    for j in range(15):
                        if results['gt_keypoints_flag'][n][j][0] == 0:
                            keypoints[n][j] = [0 for _ in range(11)]
                            
                results[key] = keypoints
        return results
                
    def __repr__(self):
        repr_str = super(AugCrop, self).__repr__()[:-1] + ', '
        repr_str += f'kpt_clip_border={self.kpt_clip_border})'
        return repr_str
