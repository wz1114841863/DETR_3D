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
    x, y,做相应的变换处理
还需要对 X, Y, Z, fx, fy, cx, cy 等做处理
对2d坐标做变换不影响深度Z
"""


@PIPELINES.register_module()
class AugRandomFlip(MMDetRandomFlip):
    """随机翻转
    分为两种情况：
        COCO:
        MuCo:
    不影响fx, fy, cx, cy
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

        注: 这里仅翻转了2d坐标中的x和交换了对应的关键点坐标, 未针对vis=0, 1, 2做处理。
        但并未修改3d 坐标中的X, Y等其他信息
        """
        # 翻转所有关键点
        keypoints = results['gt_keypoints'].copy()
        # 交换坐标, 翻转标志位 
        keypoints[:, :, 0] = \
            results['img_shape'][1] - 1 - keypoints[:, :, 0]
        keypoints[:, :, :] = keypoints[:, FLIP_ORDER, :]

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
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxs = results[key]  # [person_num, 4]")
            assert bboxs.shape[1] == 4, "the shape of bbox isn't 4."
            flip_bboxs = bboxs.copy()
            flip_bboxs[:, [0, 2]] = img_shape[1] - 1 - bboxs[:, [0, 2]]
            flip_bboxs[:, [0, 2]] = flip_bboxs[:, [2, 0]]
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
    MuCo的area是由bbox乘积而来，旋转之后，bbox重新计算，area也需重新计算
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
            # vis_flag = results['gt_vis_flag'].copy()
            for i in range(len(keypoints)):  # [person_num, 15, 11]
                keypoints[i][:, :2] = \
                    self._rotate_coordinate(keypoints[i][:, :2], self.M)
                
            # 越界处理
            img_shape = results['img_shape']
            keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, img_shape[1] - 1)
            keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, img_shape[0] - 1)
            results[key] = keypoints
        return results
    
    def _rotate_bbox(self, results):
        """旋转bbox

        Args:
            results (_type_): _description_
        """
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxs = results[key].copy()
            for i in range(len(bboxs)):  # [person_num, 4]
                bbox = bboxs[i]
                top_left = np.asarray((min(bbox[0], bbox[2]), min(bbox[1], bbox[3])), dtype=np.float32)
                top_right = np.asarray((max(bbox[0], bbox[2]), min(bbox[1], bbox[3])), dtype=np.float32)
                bottom_left = np.asarray((min(bbox[0], bbox[2]), max(bbox[1], bbox[3])), dtype=np.float32)
                bottom_right = np.asarray((max(bbox[0], bbox[2]), max(bbox[1], bbox[3])), dtype=np.float32)
                coords = np.stack([top_left, top_right, bottom_left, 
                                    bottom_right], axis=0)
                coords = self._rotate_coordinate(coords, self.M)
                assert coords.shape == (4, 2), \
                    f"error. the shape of coords is: {coords.shape}"
                
                # 重新框定bbox: x1, y1, x2, y2
                [min_x, min_y] = np.min(coords, axis=0)
                [max_x, max_y] = np.max(coords, axis=0)
                # 由于图片进行填充，按理bbox并不会越界。
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
    存在的问题：
        如果keypoints被裁剪，而COCO数据集的areas未做对应处理
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
            keypoints[:, :, :2] = \
                    keypoints[:, :, :2] * results['scale_factor'][:2]
            # 关键点越界处理，按理这里关键点并不会越界
            if self.keypoint_clip_border:
                img_shape = results['img_shape']
                keypoints[..., 0] = np.clip(keypoints[..., 0], 0, img_shape[1])
                keypoints[..., 1] = np.clip(keypoints[..., 1], 0, img_shape[0])
        
        results[key] = keypoints
    
    def _resize_areas(self, results):
        """由于img 和 keypoints 全都进行了resize，需要对areas也进行resize
        gt_
        分为两种情况：
            COCO:  直接放缩
            MUCO:  留着最后重新计算
        """
        dataset_type = results['dataset']
        if dataset_type == "COCO":
            areas = results['gt_areas'].copy()
            areas = areas * results['scale_factor'][0] \
                * results['scale_factor'][1]
            results['gt_areas'] = areas

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
        self._resize_areas(results)
        
        return results

    def __repr__(self):
        repr_str = super(AugResize, self).__repr__()[:-1] + ', '
        repr_str += f'keypoint_clip_border={self.keypoint_clip_border})'
        return repr_str


@PIPELINES.register_module()
class AugCrop(MMDetRandomCrop):
    """随机裁剪 img, bbox, keypoints
    存在的问题：
        如果keypoints被裁剪，而COCO数据集的areas未做对应处理
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
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxes = results[key].copy()  # [num_persons, 4]
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                    dtype=np.float32)
            bboxes = bboxes - bbox_offset  # 注意这里是减去偏移。
            # 判断是否超出范围, 这里要确保bbox中的数据满足[x1, y1, x2, y2]
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            if (not valid_inds.any() and not allow_negative_crop):
                print(f"{results['img_prefix']} 裁剪后不包含有效的bbox")
                return None
            
            # 留取有效的bbox[N, 4] 和 keypoints[N, J, 11]
            results[key] = bboxes[valid_inds]
            keypoints = results["gt_keypoints"].copy()
            results["gt_keypoints"] = keypoints[valid_inds]
            labels = results['gt_labels'].copy()
            results['gt_labels'] = labels[valid_inds]
            areas = results['gt_areas'].copy()
            results['gt_areas'] = areas[valid_inds]
            assert len(results[key]) == len(results["gt_keypoints"]), \
                "bbox长度与keypoints长度不一致"
            
            for key in results.get('keypoint_fields', ['gt_keypoints']):
                kpt_offset = np.array([offset_w, offset_h], dtype=np.float32)
                keypoints = results[key].copy()
                for i in range(len(keypoints)):
                    keypoints[i][:, :2] = \
                        keypoints[i][:, :2] - kpt_offset  # x, y - oofset_x, offset_y
                    # 越界处理
                    valid_inds = (keypoints[i][:, 0] >= 0.0) & (keypoints[i][:, 1] >= 0.0)
                    keypoints[i][~ valid_inds] = np.asarray([0.0 for _ in range(11)], dtype=np.float32)
                    
                assert keypoints.shape[-2:] == (15, 11), \
                    f"error in process keypoints, keypoints.shape: {keypoints.shape}"
                results[key] = keypoints
                
        return results
                
    def __repr__(self):
        repr_str = super(AugCrop, self).__repr__()[:-1] + ', '
        repr_str += f'kpt_clip_border={self.kpt_clip_border})'
        return repr_str


@PIPELINES.register_module()
class AugPostProcess():
    """
    作用如下：
        判断数据的有效性:
            判断gt的长度是否相同, bboxes, keypoints, areas, labels,
            判断某个人的十五个关键点是否可见，vis.sum > 0
        后处理：
            处理areas: 
            处理scale_factor:
            处理深度:
    """
    def __init__(self, 
                    with_proc_areas=True,
                    with_proc_length=True,
                    with_proc_kpts=True,
                    with_proc_coord=True,):
        self.with_proc_areas = with_proc_areas
        self.with_judge_length = with_proc_length
        self.with_proc_kpts = with_proc_kpts
        self.with_proc_coord = with_proc_coord

    @staticmethod
    def _proc_coord(results):
        img_shape = results['img'].shape
        results['img_shape']= img_shape

        keypoints = results['gt_keypoints'].copy()
        keypoints[..., 0] = np.clip(keypoints[..., 0], 0, img_shape[1] - 1)
        keypoints[..., 1] = np.clip(keypoints[..., 1], 0, img_shape[0] - 1)
        results['gt_keypoints'] = keypoints

        bboxes = results['gt_bboxes']
        bboxes[..., 0::2] = np.clip(bboxes[..., 0::2], 0, img_shape[1] - 1)
        bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, img_shape[0] - 1)  
        results['gt_bboxes'] = bboxes

        return results

    @staticmethod
    def _proc_areas(results):
        dataset = results['dataset']
        if dataset == "MUCO":
            bboxs = results['gt_bboxes'].copy()
            areas = []
            for i in range(len(bboxs)):
                [left_top_x, left_top_y, right_bottom_x, right_bottom_y] = bboxs[i]
                area = np.abs((right_bottom_x - left_top_x) * (right_bottom_y - left_top_y))
                areas.append(area)
            areas = np.array(areas, dtype=np.float32)
            results['gt_areas'] = areas
        elif dataset == "COCO":
            """关键点和bbox可能有裁剪，但对areas没有做对应处理，仅保证长度相同"""
            pass
        else:
            raise NotImplemented(f"未知的dataset: {dataset}")
        
        return results

    @staticmethod
    def _proc_kpts(results):
        """
            根据vis标志位，判断是否十五个关键点vis全为零，如果是，删除该目标
        """
        keypoints = results['gt_keypoints'].copy()
        valid_flag = keypoints[..., 3]
        bboxes = results['gt_bboxes'].copy()
        areas = results['gt_areas'].copy()
        labels = results['gt_labels'].copy()

        new_kpts = []
        new_bboxes = []
        new_areas = []
        new_labels = []

        for i in range(len(keypoints)):  # num_gts
            kpt_valid_flag = valid_flag[i] > 0
            assert kpt_valid_flag.shape[0] == 15
            if kpt_valid_flag.sum() == 0:
                continue
            elif not (areas[i] > 0).all():
                continue
            elif not ((bboxes[i][2] > bboxes[i][0]) & (bboxes[i][3] > bboxes[i][1])):
                continue
            else:
                new_kpts.append(keypoints[i])
                new_bboxes.append(bboxes[i])
                new_areas.append(areas[i])
                new_labels.append(labels[i])
        
        results['gt_keypoints'] = np.array(new_kpts, dtype=np.float32)
        results['gt_bboxes'] = np.array(new_bboxes, dtype=np.float32)
        results['gt_areas'] = np.array(new_areas, dtype=np.float32)
        results['gt_labels'] = np.array(new_labels)
        return results

    @staticmethod
    def _proc_scale(results):
        """重新计算scale_factor
        旋转，resize，crop均导致图片大小发生变化
        """     
        h, w = results['img_shape'][:2]
        ori_h, ori_w = results['ori_shape'][:2]
        scale_w = w / ori_w
        scale_h = h / ori_h
        scale_factor = [scale_w, scale_h, scale_w, scale_h]
        results['scale_factor'] = scale_factor
        
        return results

    @staticmethod
    def _proc_depths(results):
        """将关键点坐标与深度分离，同时对深度按照SMAP格式进行处理
        depth = depth / scale_x / f_x
        """
        keypoints = results['gt_keypoints'].copy()
        if keypoints.shape[0] != 0:
            kpt_coords = keypoints[:, :, :2]  # (x, y), [num_gts, 15, 11]
            vis_flag = keypoints[:, :, 3]  # (vis)
            kpt = np.concatenate((kpt_coords, vis_flag[..., None]), -1)
            results['gt_keypoints'] = kpt
            
            depth = keypoints[:, :, -5:]  # [Z, fx, fy, cx, cy]. [num_gts, 15, 5]
            depth_Z = np.zeros(vis_flag.shape, dtype=np.float32)
            scale_w = results['scale_factor'][0]
            depth_Z[vis_flag > 0] = depth[vis_flag > 0][:, 0] / scale_w / \
                depth[vis_flag > 0][:, 1]  # [num_gts, 15]
            # 使用相对深度进行监督
            root_idx = 2
            root_depth = depth_Z[:, root_idx].copy()
            root_depth = root_depth[:, np.newaxis]
            depth_Z = depth_Z - root_depth
            depth_Z[:, root_idx] = root_depth.squeeze(-1)
            results['gt_depths'] = depth_Z
        else:
            results['gt_depths'] = 0
        
        return results

    @staticmethod
    def _proc_length(results):
        len_bboxes = len(results['gt_bboxes'])
        if len_bboxes == 0:
            return None
        
        # 验证：
        assert len_bboxes == len(results['gt_areas']), \
            f"len_bboxes: {len_bboxes}, len_areas: {len(results['gt_areas'])}"

        assert len_bboxes == len(results['gt_keypoints']), \
            f"len_bboxes: {len_bboxes}, len_keypoints: {len(results['gt_keypoints'])}"

        assert len_bboxes == len(results['gt_labels']), \
            f"len_bboxes: {len_bboxes}, len_labels: {len(results['gt_labels'])}"
            
        assert len_bboxes == len(results['gt_depths']), \
            f"len_bboxes: {len_bboxes}, len_depths: {len(results['gt_depths'])}"
            
        return results

    def __call__(self, results):
        if self.with_proc_coord:
            results = self._proc_coord(results)
        if self.with_proc_areas:
            results = self._proc_areas(results)
        if self.with_proc_kpts:
            results = self._proc_kpts(results)
            results = self._proc_scale(results)
            results = self._proc_depths(results)
        if self.with_judge_length:  
            results = self._proc_length(results)

        return results 