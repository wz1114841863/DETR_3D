# Document function description
#   模仿SMAP的数据格式，加载已经修改格式之后的COCO和MuCo数据集。
import warnings
from mmdet.datasets import CustomDataset
import mmcv
import numpy as np
import json
import scipy.io as scio
from scipy.optimize import linear_sum_assignment

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

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        ann_info = self.get_ann_info(idx)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

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

    def _calc_iou(self, bbox1, bbox2):
        """calculate the IOU
        Args:
            bbox1 (_type_): 
                [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            bbox2 (_type_): 
                [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        """
        x1_min, y1_min, x1_max, y1_max = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x2_min, y2_min, x2_max, y2_max = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        assert x1_min < x1_max and y1_min < y1_max and \
            x2_min < x2_max and y2_min < y2_max, \
            f"error. bbox 中坐标范围不合理"
        # 计算两个矩形框面积
        area1 = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
        area2 = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
        # 计算交点的坐标（xleft, yleft, xright, yright)
        xmin = np.max([x1_min, x2_min])
        ymin = np.max([y1_min, y2_min])
        xmax = np.min([x1_max, x2_max])
        ymax = np.min([y1_max, y2_max])
        inter_bbox = [xmin, ymin, xmax, ymax]
        # 计算交集面积
        inter_area = np.max([0, xmax - xmin]) * np.max([0, ymax - ymin])  # 可能没有交集，此时面积为0
        if inter_area == 0:
            return None, 0
        # 计算iou
        # A = area(bbox1), B = area(bbox2), C = （area(bbox1) ∩ area(bbox2）
        # iou = C / (A + B - C)
        iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 防止分母为零

        return inter_bbox, iou

    def _get_pred_indexs_by_iou(self, gt_bboxs, pred_bboxs):
        """根据iou来计算与gt最为匹配的输出
        可能存在问题: 
            bbox重叠之类的影响
            不存在匹配的bbox
        Args:
            gt_bboxs (numpy.array): 
                gt bboxs, [num_gts, 4]
                [top_left_x, top_left_y, weight, height]
            pred_bboxes (numpy.array): 
                pred bboxs, [num_preds, 5]
                [top_left_x, top_left_y, bottom_right_x, bottom_right_y, scores]
        """
        pred_indexs = []
        for i in range(len(gt_bboxs)):
            index = -1
            iou = 0.
            for j in range(len(pred_bboxs)):
                _, iou_tmp = self._calc_iou(gt_bboxs[i], pred_bboxs[j][:4])
                if (iou_tmp != 0 and iou_tmp > iou and j not in pred_indexs):
                    iou = iou_tmp
                    index = j
            if index == -1:
                # TODO bug 待修复
                # print(f"index == -1")
                index = 0
            # assert index != -1, f"没有一个bbox与之匹配"
            pred_indexs.append(index)
            
        return pred_indexs

    def _get_pred_indexs_by_scores(self, gt_bboxs, pred_bboxs):
        """
        首先选取置信度最高的前num_gts个, 然后根据iou进行匹配   
        前提是返回的置信度需要从高到底排列    
        """
        num_gts = len(gt_bboxs)
        # 选取前num_gts个
        pred_bboxs_tmp = pred_bboxs[:num_gts, :4].copy()
        # 进行iou匹配
        iou_matrix = np.zeros((num_gts, num_gts))
        for i in range(num_gts):
            for j in range(num_gts):
                _, iou_tmp = self._calc_iou(gt_bboxs[i], pred_bboxs_tmp[j])
                assert iou_tmp <= 1 and iou_tmp >= 0, \
                    f"iou 应该在0-1范围内"
                iou_matrix[i][j] = iou_tmp

        not_iou_matrix = 1 - iou_matrix
        matched_row_inds, matched_col_inds = linear_sum_assignment(not_iou_matrix)
        return matched_col_inds

    def _proc_gt_bodys(self, gt_bodys):
        """添加中心点

        Args:
            gt_bodys (_type_): [num_gts, 15, 11]
        Return:
            gt_bodys (_type_): [num_gts, 16, 11]
        """
        num_gts = gt_bodys.shape[0]
        new_gt_bodys = np.zeros((num_gts, 16, 11))
        for i in range(num_gts):
            new_gt_bodys[i][:15, :] = gt_bodys[i][:, :]
            valid_index = gt_bodys[i][:, 3] > 0
            if valid_index.sum() != 0:
                tmp = gt_bodys[i][valid_index]  # [valid_index.sum(), 11]
                new_gt_bodys[i][15] = np.sum(tmp, 0) / valid_index.sum()
        
        return new_gt_bodys 
    
    def _proc_gt_bboxs(self, gt_bboxs):
        """
            处理gt bboxs 的格式
            从[x, y, w, h] -> [x1, y1, x2, y2]
        """
        gt_bboxs[:, 2] += gt_bboxs[:, 0]
        gt_bboxs[:, 3] += gt_bboxs[:, 1]

        return gt_bboxs

    def _proc_2d(self, pred_kpts, pred_depths, pred_scores, scale_dict):
        """生成2d坐标和相对于骨盆点的深度

        Args:
            pred_kpts (numpy.ndarray): [num_gts, 15, 2]
            pred_depths (numpy.ndarray): [num_gts, 16]
            pred_scores (numpy.ndarray): [num_gts, 1]
            scale_dict (dict): scale
        """
        assert len(pred_kpts) == len(pred_depths), \
            f"关键点长度与depth长度不同"
        assert len(pred_kpts) == len(pred_scores), \
            f"关键点长度与scores长度不同"        
        num_gts = len(pred_depths)
        root_idx = 2  # 骨盆点
        pred_bodys_2d = np.zeros((num_gts, 15, 4))
        pred_rdepths = np.zeros((num_gts, ))  # (骨盆点深度, )
        for i in range(num_gts):
            # 根据smap公式计算Z，返回时已经将2d坐标乘以scale，detZ也已经乘以scale
            # 所以这里仅乘以了 fx
            kpt_abs_depth = pred_depths[i] * scale_dict['f_x']
            # 骨盆点深度
            root_abs_depth = kpt_abs_depth[root_idx].copy()
            # 其余点相对于骨盆点的深度
            kpt_rel_root_depth = kpt_abs_depth - kpt_abs_depth[root_idx]
            pred_bodys_2d[i][:, :2] = pred_kpts[i]
            pred_bodys_2d[i][:, 2] = kpt_rel_root_depth
            pred_bodys_2d[i][:, 3] = pred_scores[i]
            pred_rdepths[i] = root_abs_depth
            
        return pred_bodys_2d, pred_rdepths
    
    def _back_projection(self, x, d, K):
        """
        Back project 2D points x(2xN) to the camera coordinate and ignore distortion parameters.
        :param x: 2*N
        :param d: real depth of every point
        :param K: camera intrinsics
        :return: X (2xN), points in 3D
        P3D.x = (x_2d - cx_d) * depth(x_2d,y_2d) / fx_d
        P3D.y = (y_2d - cy_d) * depth(x_2d,y_2d) / fy_d
        P3D.z = depth(x_2d,y_2d)

        已验证, 与下列方式计算结果相同:
            iK = np.linalg.inv(K)
            tmp2d = np.array(x[i][0], x[i][1], 1]).reshape([3,1])
            (d * iK @ tmp2d)
        """
        X = np.zeros((len(d), 3), np.float)
        X[:, 0] = (x[:, 0] - K[0, 2]) * d / K[0, 0]
        X[:, 1] = (x[:, 1] - K[1, 2]) * d / K[1, 1]
        X[:, 2] = d
        return X

    def _get_3d_points(self, pred_bodys, root_depth, K, root_n=2):
        bodys_3d = np.zeros(pred_bodys.shape, np.float)
        bodys_3d[:, :, 3] = pred_bodys[:, :, 3]
        for i in range(len(pred_bodys)):
            pred_bodys[i][:, 2] += root_depth[i]
            bodys_3d[i][:, :3] = self._back_projection(pred_bodys[i][:, :2], pred_bodys[i][:, 2], K)
        return bodys_3d

    def _proc_3d(self, pred_bodys_2d, pred_rdepths, scale):
        """求相对深度和

        Args:
            pred_bodys_2d (numpy.ndarray): [num_gts, 15, 4]
            pred_scores (numpy.ndarray): _description_
            scale (dict): scale
        """
        coords_2d = pred_bodys_2d.copy()
        root_depth = pred_rdepths.copy()
        K = np.asarray([[scale['f_x'], 0, scale['cx']], [0, scale['f_y'], scale['cy']], [0, 0, 1]])
        # 获取3d坐标
        pred_bodys_3d = self._get_3d_points(coords_2d, root_depth, K)
        return pred_bodys_3d
        
    def evaluate(self, 
                results, 
                output_save_path,
                mat_save_path,
                save_json=True,
                save_mat=True,):
        """生成用于进行eval的json文件, 参考SMAP。

        Args:
            results (_type_): 网络输出结果
            output_save_path: json保存路径
            mat_save_path:  mat保存路径
            save_json: 是否保存json
            save_mat: 是否保存mat

            3d_pairs has items like{'pred_2d':[[x,y,detZ,score]...], 
                                    'gt_2d':[[x,y,Z,visual_type]...],
                                    'pred_3d':[[X,Y,Z,score]...], 
                                    'gt_3d':[[X,Y,Z]...],
                                    'root_d': (abs depth of root (float value) pred by network),
                                    'image_path': relative image path}
        """
        assert len(self.data_infos) == len(results), \
            f"len(anno) != len(results), length of anno is {len(anno)}"
        output = dict()
        output['model_pattern'] = self.__class__.__name__
        output['3d_pairs'] = []
        # TODO 查看eval时是否看顺序读取数据集，否则下面代码逻辑错误
        for i in range(len(results)):
            anno = self.data_infos[i]
            result = results[i]
            # 取出result中对应数据
            bboxs, kpts, depths = result[0][0], result[1][0], result[2][0]
            assert bboxs.shape == (30, 5), \
                f"error. bboxs.shape:{bboxs.shape}"
            assert kpts.shape == (30, 15, 2), \
                f"error. kpts.shape:{kpts.shape}"
            assert depths.shape == (30, 15), \
                f"error. depths.shape:{depths.shape}"
            # gt 处理
            gt_bodys = []
            gt_bboxs = []
            anno_bodys = np.array(anno['bodys'])
            for j in range(len(anno_bodys)):
                valid_num = anno_bodys[j][:, 3] > 0  # [15, ]
                if valid_num.sum() > 0:
                    gt_bodys.append(anno['bodys'][j])
                    gt_bboxs.append(anno['bboxs'][j])
                else:
                    print(f"{anno['img_paths']} 中包含有无效bbox和body")
            
            gt_bodys = np.array(gt_bodys)  # [num_gts, 15, 11]
            gt_bodys = self._proc_gt_bodys(gt_bodys)  # [num_gts, 16, 11]
            assert gt_bodys.shape[-2:] == (16, 11) , f"gt_bodys.shape: {gt_bodys.shape}"
            gt_bboxs = np.array(gt_bboxs)
            gt_bboxs = self._proc_gt_bboxs(gt_bboxs)
            assert gt_bboxs.shape[-1] == 4, f"gt_bboxs.shape: {gt_bboxs.shape}"

            num_gts = len(gt_bboxs)
            if num_gts == 0:
                print(f"{anno['img_paths']} 中gt个数为0")
                continue

            scale_dict = dict()
            scale_dict['img_w'] = anno['img_width']
            scale_dict['img_h'] = anno['img_height']
            scale_dict['f_x'] = gt_bodys[0, 0, 7]
            scale_dict['f_y'] = gt_bodys[0, 0, 8]
            scale_dict['cx'] = gt_bodys[0, 0, 9]
            scale_dict['cy'] = gt_bodys[0, 0, 10]
            # 获取与gt数目相等的preds
            # FIXME 两种方法：
            # 利用 iou 与 利用置信度 存在差别
            pred_indexs = self._get_pred_indexs_by_scores(gt_bboxs, bboxs)
            pred_indexs = np.array(pred_indexs)
            assert len(pred_indexs) == len(gt_bboxs), \
                f"error. Unequal length."
            pred_bboxes, pred_scores = bboxs[pred_indexs][:, :4], bboxs[pred_indexs][:, 4][:, None]  # [num_gts, 4], [num_gts, 1]
            pred_kpts = kpts[pred_indexs]  # [num_gts, 15, 2]
            pred_depths = depths[pred_indexs]  # [num_gts, 16]
            # 对2d坐标和3d坐标进行处理
            pred_bodys_2d, pred_rdepths = self._proc_2d(pred_kpts, pred_depths, pred_scores, scale_dict)  
            # pred_bodys_2d: (x, y, relative depth, scores), pred_rdepths: (absolute depth) 骨盆点的绝对深度
            pred_bodys_3d = self._proc_3d(pred_bodys_2d, pred_rdepths, scale_dict)  # (X, Y, absolute depth, scores)
            # 检验长度
            assert len(gt_bodys) == num_gts
            assert pred_bodys_2d.shape == (num_gts, 15, 4)
            assert pred_bodys_3d.shape == (num_gts, 15, 4)
            assert pred_rdepths.shape[0] == num_gts

            pair = dict()
            pair['pred_2d'] = pred_bodys_2d.tolist()
            pair['pred_3d'] = pred_bodys_3d.tolist()
            pair['root_d'] = pred_rdepths.tolist()
            pair['image_path'] = anno['img_paths']
            pair['gt_3d'] = gt_bodys[:, :, 4:].tolist()
            pair['gt_2d'] = gt_bodys[:, :, :4].tolist()

            output['3d_pairs'].append(pair)

        if save_json:
            file_path = output_save_path + 'output.json'
            with open(file_path, 'w+') as fp:
                json.dump(output, fp, indent=4)
            print(f"\n output结果写入至: {file_path}")

        if save_mat:
            self._save_result_to_mat(output, mat_save_path)
        return
    
    def _save_result_to_mat(self, output, mat_save_path):
        """
            将eval的结果存储为mat格式进行保存
        """
        pairs_3d = output['3d_pairs']

        pose3d = dict()
        pose2d = dict()

        for i in range(len(pairs_3d)):
            img_path = pairs_3d[i]['image_path']
            idx = img_path.index('TS')
            name = img_path[idx:]
            pred_3ds = np.array(pairs_3d[i]['pred_3d'])  # [num_gt, 15, 4] 
            pred_2ds = np.array(pairs_3d[i]['pred_2d'])  # [num_gts, 15, 4]
            pose3d[name] = pred_3ds * 10 # nH x 15 x 4 
            pose3d[name][:, :, 3] /= 10
            pose2d[name] = pred_2ds # nH x 15 x 4

        # 保存结果
        # pose3d_save_path = mat_save_path + 'pose3d.mat'
        # pose2d_save_path = mat_save_path + 'pose2d.mat'
        pose3d_save_path = './pose3d.mat'
        pose2d_save_path = './pose2d.mat'
        scio.savemat(pose3d_save_path, {'preds_3d_kpt':pose3d})
        scio.savemat(pose2d_save_path, {'preds_2d_kpt':pose2d})
        print(f"pose3d mat 数据保存至: {pose3d_save_path}")
        print(f"pose2d mat 数据保存至: {pose2d_save_path}")

    def __repr__(self):
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                    f'with number of images {len(self)}, \n')
        return result

if __name__ == '__main__':
    pass

