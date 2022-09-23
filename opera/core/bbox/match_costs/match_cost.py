# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import numpy as np

from .builder import MATCH_COST


@MATCH_COST.register_module()
class KptL1Cost(object):
    """KptL1Cost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import KptL1Cost
        >>> import torch
        >>> self = KptL1Cost()
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with normalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].

        Returns:
            torch.Tensor: kpt_cost value with weight.
        """
        kpt_cost = []
        for i in range(len(gt_keypoints)):
            kpt_pred_tmp = kpt_pred.clone()  # [300 ,17, 2]
            valid_flag = valid_kpt_flag[i] > 0  # [17, ]
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(
                -1).expand_as(kpt_pred_tmp)  # [300 ,17, 2]
            kpt_pred_tmp[~valid_flag_expand] = 0  # 去除无效关键点
            cost = torch.cdist(  # [300, 1]
                kpt_pred_tmp.reshape(kpt_pred_tmp.shape[0], -1),  # [300, 34]
                gt_keypoints[i].reshape(-1).unsqueeze(0),  # [1, 34]
                p=1)
            avg_factor = torch.clamp(valid_flag.float().sum() * 2, 1.0)  # float
            cost = cost / avg_factor  # [300, 1],
            kpt_cost.append(cost)
        kpt_cost = torch.cat(kpt_cost, dim=1)  # [300, 5]
        return kpt_cost * self.weight


@MATCH_COST.register_module()
class OksCost(object):
    """OksCost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import OksCost
        >>> import torch
        >>> self = OksCost()
    """

    def __init__(self, num_keypoints=17, weight=1.0):
        self.weight = weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26,
                .25, .25,
                .35, .35,
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array([
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89,
                .79, .79], dtype=np.float32) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag, gt_areas):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].
            gt_areas (Tensor): Ground truth mask areas. Shape [num_gt,].

        Returns:
            torch.Tensor: oks_cost value with weight.
        """
        sigmas = torch.from_numpy(self.sigmas).to(kpt_pred.device)
        variances = (sigmas * 2)**2

        oks_cost = []
        assert len(gt_keypoints) == len(gt_areas)
        for i in range(len(gt_keypoints)):
            squared_distance = \
                (kpt_pred[:, :, 0] - gt_keypoints[i, :, 0].unsqueeze(0)) ** 2 + \
                (kpt_pred[:, :, 1] - gt_keypoints[i, :, 1].unsqueeze(0)) ** 2  # [300, 17]
            vis_flag = (valid_kpt_flag[i] > 0).int()  # [17, ]
            vis_ind = vis_flag.nonzero(as_tuple=False)[:, 0]  # valid_kpt_index
            num_vis_kpt = vis_ind.shape[0]
            assert num_vis_kpt > 0
            area = gt_areas[i]

            squared_distance0 = squared_distance / (area * variances * 2)  # [300, 17]
            squared_distance0 = squared_distance0[:, vis_ind]  # [300, valid_kpt_index]
            squared_distance1 = torch.exp(-squared_distance0).sum(
                dim=1, keepdim=True)  # [300, 1]
            oks = squared_distance1 / num_vis_kpt  # [300, 1]
            # The 1 is a constant that doesn't change the matching, so omitted.
            oks_cost.append(-oks)
        oks_cost = torch.cat(oks_cost, dim=1)  # [300, num_gts]
        return oks_cost * self.weight


@MATCH_COST.register_module()
class DepthL1Cost(object):
    """用来计算深度Cost
    包括两个方面：
        参考点的绝对深度Cost
        十五个关键点的绝对深度Cost
    # TODO 可能需要进一步考虑和优化Cost的计算方式
    Args:
        weight (int | float, optional): loss_weight.
        
    Returns:
        torch.Tensor: depth_cost value with weight.
    """
    def __init__(self, 
                    kpt_depth_weight=1.0,
                    refer_depth_weight=1.0):
        self.kpt_depth_weight = kpt_depth_weight
        self.refer_depth_weight = refer_depth_weight
        
    def __call__(self, refer_point_depth, kpt_abs_depth, 
            gt_keypoints_depth, valid_kpt_flag, img_shape):
        """求深度损失 及其所占比例

        Args:
            refer_point_depth (Tensor): 预测的参考点的绝对深度  # [300, 1]
            kpt_abs_depth (Tensor): 预测的关键点的绝对深度  # [300, 15]
            gt_keypoints_depth (Tensor): gt_keypoints_depth [N, 15, 5], [Z, fx, fy, cx, cy]
            valid_kpt_flag (_type_): 有效的关键点标志位， vis, [N, 15, 1]
            img_shape: [h, w]
        """
        kpt_depth_cost = []
        refer_depth_cost= []
        for i in range(len(gt_keypoints_depth)):
            refer_depth_tmp = refer_point_depth.clone()
            kpt_depth_tmp = kpt_abs_depth.clone()
            valid_flag = valid_kpt_flag[i] > 0  # [15]
            
            # 计算参考点深度损失
            gt_aver_depth = torch.sum(gt_keypoints_depth[i][valid_flag][0]) / len(gt_keypoints_depth[i][valid_flag][0])
            refer_cost = torch.abs(refer_depth_tmp - gt_aver_depth)
        
            # 计算关键点深度损失
            kpt_depth_tmp[~valid_flag] = 0  # 无效深度 
            # FIXME 根据SMAP对深度进行处理
            depth = gt_keypoints_depth[i][valid_flag][0] / gt_keypoints_depth[i][valid_flag][1]  # Z / fx, [15, ]
            kpt_cost = torch.cidst(
                kpt_depth_tmp,
                depth.unsqueeze(0),  # [1, 15]
                p=1,)
            # 
            avg_factor = torch.clamp(valid_flag.float().sum(), 1.0)
            kpt_cost = kpt_cost / avg_factor
            
            refer_depth_cost.append(refer_cost)
            kpt_depth_cost.append(kpt_cost)
            
        kpt_depth_cost = torch.cat(kpt_depth_cost, dim=1)
        refer_depth_cost = torch.cat(refer_depth_cost, dim=1)
        
        return refer_depth_cost * self.refer_depth_weight, \
                kpt_depth_cost * self.kpt_depth_weight