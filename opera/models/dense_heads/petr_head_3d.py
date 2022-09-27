# Copyright (c) Hikvision Research Institute. All rights reserved.
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Linear, bias_init_with_prob, constant_init, normal_init,
                        build_activation_layer)
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.dense_heads import AnchorFreeHead

from opera.core.bbox import build_assigner, build_sampler
from opera.core.keypoint import gaussian_radius, draw_umich_gaussian
from opera.models.utils import build_positional_encoding, build_transformer
from ..builder import HEADS, build_loss


@HEADS.register_module()
class PETRHead3D(AnchorFreeHead):
    """Head of `End-to-End Multi-Person Pose Estimation with Transformers`.
        增添3D深度预测。将object query -> [300, 17 * 3]
    Args:
        num_classes (int): Number of categories excluding the background.  # PETR中仅使用了人
        in_channels (int): Number of channels in the input feature map.  # 2048
        num_query (int): Number of query in Transformer.
        num_kpt_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the keypoint regression head.
            Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): ConfigDict is used for
            building the Encoder and Decoder. Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.    # 同步进程信息
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_kpt (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_oks (obj:`mmcv.ConfigDict`|dict): Config of the
            regression oks loss. Default `OKSLoss`.
        loss_hm (obj:`mmcv.ConfigDict`|dict): Config of the
            regression heatmap loss. Default `NegLoss`.
        loss_3d_depth (obj:`mmcv.ConfigDict`|dict): Config of the
            3D Depth loss. Default `L1Loss`.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        with_kpt_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to True.
        train_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                    num_classes,
                    in_channels,
                    num_query=300,
                    num_kpt_fcs=2,
                    num_keypoints=15,
                    transformer=None,
                    sync_cls_avg_factor=True,
                    positional_encoding=dict(
                        type='SinePositionalEncoding',
                        num_feats=128,
                        normalize=True),
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=2.0),
                    loss_kpt=dict(type='L1Loss', loss_weight=70.0),
                    loss_oks=dict(type='OKSLoss', loss_weight=2.0),
                    loss_depth=dict(type='mmdet.L1Loss', loss_weight=70.0),
                    loss_hm=dict(type='CenterFocalLoss', loss_weight=4.0),
                    as_two_stage=True,
                    with_kpt_refine=True,
                    with_depth_refine=True,
                    train_cfg=dict(
                        assigner=dict(
                            type='PoseHungarianAssigner3D',
                            cls_cost=dict(type='FocalLossCost', weight=2.0),
                            kpt_cost=dict(type='KptL1Cost', weight=70.0),
                            oks_cost=dict(type='OksCost', weight=7.0),
                            depth_cost=dict(type='KptL1Cost', weight=70.0))),
                    loss_kpt_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                    loss_depth_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                    loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=70.0),
                    loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=2.0),
                    test_cfg=dict(max_per_img=100),
                    init_cfg=None,
                    **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor  # True
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_kpt['loss_weight'] == assigner['kpt_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='mmdet.PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query  # 300
        self.num_classes = num_classes  # 1
        self.in_channels = in_channels  # 2048
        self.num_kpt_fcs = num_kpt_fcs  # 2
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg  # {max_per_img: 100}
        self.fp16_enabled = False
        self.as_two_stage = as_two_stage  # True
        self.with_kpt_refine = with_kpt_refine  # True
        self.with_depth_refine = with_depth_refine  
        self.num_keypoints = num_keypoints  # 15
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage  
        # transformer: decoder, encoder, hm_encoder, refine_decoder
        
        self.loss_cls = build_loss(loss_cls)  # FocalLoss()
        self.loss_kpt = build_loss(loss_kpt)  # L1Loss()
        self.loss_kpt_rpn = build_loss(loss_kpt_rpn)  # L1Loss()
        self.loss_kpt_refine = build_loss(loss_kpt_refine)  # L1Loss()
        self.loss_oks = build_loss(loss_oks)  # OKSLoss()
        self.loss_oks_refine = build_loss(loss_oks_refine)  # OKSLoss()
        self.loss_hm = build_loss(loss_hm)  # CenterFocalLoss()
        self.loss_depth = build_loss(loss_depth)  # L1Loss()
        self.loss_depth_rpn = build_loss(loss_depth_rpn)  # L1Loss()
        # self.loss_depth_refine = build_loss(loss_depth_refine)  # L1Loss()
        
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes  # 1
        else:
            self.cls_out_channels = num_classes + 1
            
        self.act_cfg = transformer.get('act_cfg',
                                        dict(type='ReLU', inplace=True))  # 'type': ReLu, 'inplace': True
        self.activate = build_activation_layer(self.act_cfg)  # ReLU(inplace=True)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        # SinePositionalEncoding(num_feats=128, temperature=10000, normalize=Ture, scale=6.283)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims  # 256
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']  # 128
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and keypoint branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)  # Linear(256, 1, bias=True)

        kpt_branch = []
        kpt_branch.append(Linear(self.embed_dims, 512))  # Linear(256, 512, bias=True) 
        kpt_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):  # 2
            kpt_branch.append(Linear(512, 512))
            kpt_branch.append(nn.ReLU())
        kpt_branch.append(Linear(512, 2 * self.num_keypoints))  # 不修改这里，直接添加新的回归分支
        kpt_branch = nn.Sequential(*kpt_branch)

        depth_branch = []
        depth_branch.append(Linear(self.embed_dims, 512))
        depth_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):  # 2
            depth_branch.append(Linear(512, 512))
            depth_branch.append(nn.ReLU())
        depth_branch.append(Linear(512, 1 + self.num_keypoints))
        depth_branch = nn.Sequential(*depth_branch)
        
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last kpt_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers  # 4

        if self.with_kpt_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)  # 4 * Linear(256, 1, bias=True)
            self.kpt_branches = _get_clones(kpt_branch, num_pred)  # 4 * kpt_branch
            self.depth_branches = _get_clones(depth_branch, num_pred)  # 4 * depth_branch
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.kpt_branches = nn.ModuleList(
                [kpt_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)  # Embedding(300, 512)
        # 
        refine_kpt_branch = []
        for _ in range(self.num_kpt_fcs):  # 2
            refine_kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
            refine_kpt_branch.append(nn.ReLU())
        refine_kpt_branch.append(Linear(self.embed_dims, 2))  # (x, y)
        refine_kpt_branch = nn.Sequential(*refine_kpt_branch)  # Linear(256, 3, bias=True)
        
        refine_depth_branch = []
        for _ in range(self.num_kpt_fcs): 
            refine_depth_branch.append(Linear(self.embed_dims, self.embed_dims))
            refine_depth_branch.append(nn.ReLU())
        refine_depth_branch.append(Linear(self.embed_dims, 1))  # (x, y)
        refine_depth_branch = nn.Sequential(*refine_depth_branch)  # Linear(256, 3, bias=True)    
        
        if self.with_kpt_refine:
            num_pred = self.transformer.refine_decoder.num_layers  # 2
            self.refine_kpt_branches = _get_clones(refine_kpt_branch, num_pred)  # 2 * refine_kpt_branch
        
        if self.with_depth_refine:  # 目前为False
            num_pred = self.transformer.refine_decoder.num_layers  # 2
            self.refine_depth_branches = _get_clones(refine_depth_branch, num_pred)
            
        self.fc_hm = Linear(self.embed_dims, self.num_keypoints)  # Linear(256, 15, bias=True)

    def init_weights(self):
        """Initialize weights of the PETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.kpt_branches:
            constant_init(m[-1], 0, bias=0)
        # initialization of keypoint refinement branch
        if self.with_kpt_refine:
            for m in self.refine_kpt_branches:
                constant_init(m[-1], 0, bias=0)
        # initialization of depth branch
        # for m in self.depth_branches:
        #     constant_init(m[-1], 0, bias=0)
        # if self.with_depth_refine:
        #     for m in self.refine_depth_branches:
        #         constant_init(m[-1], 0, bias=0)

        # initialization of depth branch
        for m in self.depth_branches:
            for i in range(len(m)):
                if type(m[i]) == nn.Linear:
                    nn.init.normal_(m[i].weight, std=0.01)
        # if self.with_depth_refine:
        #     for m in self.refine_depth_branches:
        #         for i in range(len(m)):
        #             if type(m[i]) == nn.Linear:
        #                 nn.init.normal_(m[i].weight, std=0.01)

        # initialize bias for heatmap prediction

        bias_init = bias_init_with_prob(0.1)
        normal_init(self.fc_hm, std=0.01, bias=bias_init)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            outputs_classes (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should include background.
            outputs_kpts (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, K*2].
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (N, h*w, num_class). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
            enc_outputs_kpt (Tensor): The proposal generate from the
                encode feature map, has shape (N, h*w, K*2). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
        """
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']  # (pad_h, pad_w)
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))  # img_masks: [bs, pad_h, pad_w]
        
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']  # (pad_h, pad_w)
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],  # 扩充维度
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = self.query_embedding.weight  # (300, 512)
        hs, inter_references, inter_depths, \
            init_reference, init_depth, \
                enc_outputs_class, enc_outputs_kpt, enc_outputs_depth, \
                    hm_proto, memory = \
                self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    depth_branches=self.depth_branches,
                    kpt_branches=self.kpt_branches \
                        if self.with_kpt_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches \
                        if self.as_two_stage else None,  # noqa:E501
            )  # transformer.forward() 返回的结果
        # hs: (3, 300, bs, 256), inter_references: [3, bs, 300, 30], inter_depths: [3, bs, 300, 16]
        # init_reference: (bs, 300, 30), init_depth: [bs, 300, 16]
        # enc_outputs_class: (bs, sum(h*w), 1), enc_outputs_kpt: (..., 30), enc_outputs_depth: (..., 16)
        # hm_proto: (hm_memory:[bs, h1, w1, 256], mlvl_masks[0]:[bs, h1, w1]), memory: (sum(h*w), bs, 256)
        hs = hs.permute(0, 2, 1, 3)  # (3, bs, 300, 256)
        enc_outputs_kpt = enc_outputs_kpt.sigmoid()  # inf -> 1.0 防止损失变为nan
        outputs_classes = []  # len = 3, (bs, 300, 1)
        outputs_kpts = []  # len = 3, (bs, 300, 46)
        outputs_depths = []  # len = 3
        
        for lvl in range(hs.shape[0]):  # 3
            if lvl == 0:
                reference_kpt = init_reference  # [bs, 300, 30]
                reference_depth = init_depth
            else:
                reference_kpt = inter_references[lvl - 1]  # [bs, 300, 30]
                reference_depth = inter_depths[lvl - 1]
            
            # class
            outputs_class = self.cls_branches[lvl](hs[lvl])  # [bs, 300, 1]
            # keypoint
            reference_kpt = inverse_sigmoid(reference_kpt)  # [bs, 300, 30]
            tmp_kpt_coord = self.kpt_branches[lvl](hs[lvl])  # [bs, 300, 30]
            tmp_kpt_coord += reference_kpt  # [bs, 300, 30]
            outputs_kpt = tmp_kpt_coord.sigmoid()
            # depth
            tmp_kpt_depth = self.depth_branches[lvl](hs[lvl])  # [bs, 300, 1 + 15]
            tmp_kpt_depth += reference_depth  # [bs, 300, 1 + 15]
            outputs_depth = tmp_kpt_depth

            outputs_classes.append(outputs_class)
            outputs_kpts.append(outputs_kpt)
            outputs_depths.append(outputs_depth)
            
        outputs_classes = torch.stack(outputs_classes)  # (3, bs, 300, 1)
        outputs_kpts = torch.stack(outputs_kpts)  # (3, bs, 300, 30)
        outputs_depths = torch.stack(outputs_depths)  # (3, bs, 300, 1 + 15)

        if hm_proto is not None:
            # get heatmap prediction (training phase)
            hm_memory, hm_mask = hm_proto  # [bs, h, w, 256], [bs, h, w]
            hm_pred = self.fc_hm(hm_memory)  # [bs, h, w, 15]
            hm_proto = (hm_pred.permute(0, 3, 1, 2), hm_mask)  # ([bs, 15, h, w], [bs, h, w])

        if self.as_two_stage:
            return outputs_classes, outputs_kpts, outputs_depths, \
                enc_outputs_class, enc_outputs_kpt, enc_outputs_depth, \
                hm_proto, memory, mlvl_masks
        else:
            raise RuntimeError('only "as_two_stage=True" is supported.')

    def forward_refine(self, memory, mlvl_masks, refine_targets, losses,
                        img_metas):
        """Forward function.

        Args:
            mlvl_masks (tuple[Tensor]): The key_padding_mask from
                different level used for encoder and decoder,
                each is a 3D-tensor with shape (bs, H, W).
            losses (dict[str, Tensor]): A dictionary of loss components.
            img_metas (list[dict]): List of image information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        area_targets, kpt_preds, kpt_targets, kpt_weights, \
            depth_preds, depth_targets, depth_weights = refine_targets  # 打包的数据
        
        pos_inds = kpt_weights.sum(-1) > 0
        if pos_inds.sum() == 0:
            pos_kpt_preds = torch.zeros_like(kpt_preds[:1])
            pos_img_inds = kpt_preds.new_zeros([1], dtype=torch.int64)
        else:
            pos_kpt_preds = kpt_preds[pos_inds]
            pos_img_inds = (pos_inds.nonzero() / self.num_query).squeeze(1).to(
                torch.int64)  # (100)

        hs, init_reference, inter_references = self.transformer.forward_refine(
            mlvl_masks,
            memory,
            pos_kpt_preds.detach(),  # 阻断反向传播
            pos_img_inds,
            kpt_branches=self.refine_kpt_branches if self.with_kpt_refine else None,  # noqa:E501
        )
        # hs: [num_dec, 15, num_gts, 256], init_reference:  [num_gts, 15, num_dec],
        # inter_references: [num_dec, num_gts, 15, 2]    
        hs = hs.permute(0, 2, 1, 3)  # [2, num_gts, 15, 256]
        outputs_kpts = []
        
        for lvl in range(hs.shape[0]):  # 2
            if lvl == 0:
                reference = init_reference  # [num_gts, 17, 2]
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)  # [num_gts, 15, 2]
            tmp_kpt = self.refine_kpt_branches[lvl](hs[lvl])  # [num_gts, 15, 2]
            assert reference.shape[-1] == 2
            tmp_kpt += reference
            outputs_kpt = tmp_kpt.sigmoid()  # [num_gts, 17, 2]
            outputs_kpts.append(outputs_kpt)
            
        outputs_kpts = torch.stack(outputs_kpts)  # [2, num_gts, 17, 2]
        

        if not self.training:
            return outputs_kpts

        batch_size = mlvl_masks[0].size(0)
        factors = []
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            factor = mlvl_masks[0].new_tensor(
                [img_w, img_h, img_w, img_h],
                dtype=torch.float32).unsqueeze(0).repeat(self.num_query, 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        factors = factors[pos_inds][:, :2].repeat(1, kpt_preds.shape[-1] // 2)

        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        num_total_pos = kpt_weights.new_tensor([outputs_kpts.size(1)])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        pos_kpt_weights = kpt_weights[pos_inds]
        pos_kpt_targets = kpt_targets[pos_inds]
        pos_kpt_targets_scaled = pos_kpt_targets * factors
        pos_areas = area_targets[pos_inds]
        pos_valid = kpt_weights[pos_inds, 0::2]
        for i, kpt_refine_preds in enumerate(outputs_kpts):
            if pos_inds.sum() == 0:
                loss_kpt = loss_oks = kpt_refine_preds.sum() * 0
                losses[f'd{i}.loss_kpt_refine'] = loss_kpt
                losses[f'd{i}.loss_oks_refine'] = loss_oks
                continue
            # kpt L1 Loss
            pos_refine_preds = kpt_refine_preds.reshape(
                kpt_refine_preds.size(0), -1)  # [num_gts, 30]
            loss_kpt = self.loss_kpt_refine(
                pos_refine_preds,
                pos_kpt_targets,
                pos_kpt_weights,
                avg_factor=num_valid_kpt)
            losses[f'd{i}.loss_kpt_refine'] = loss_kpt
            # kpt oks loss
            pos_refine_preds_scaled = pos_refine_preds * factors
            assert (pos_areas > 0).all()
            loss_oks = self.loss_oks_refine(
                pos_refine_preds_scaled,
                pos_kpt_targets_scaled,
                pos_valid,
                pos_areas,
                avg_factor=num_total_pos)
            losses[f'd{i}.loss_oks_refine'] = loss_oks
            
        return losses

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                        x,
                        img_metas,
                        dataset,
                        gt_bboxes,
                        gt_labels=None,
                        gt_keypoints=None,
                        gt_areas=None,
                        gt_bboxes_ignore=None,
                        proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_keypoints (list[Tensor]): Ground truth keypoints of the image,
                shape (num_gts, K*3).
            gt_areas (list[Tensor]): Ground truth mask areas of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        memory, mlvl_masks = outs[-2:]
        outs = outs[:-2]

        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_keypoints, gt_areas, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, 
                gt_keypoints, gt_areas, dataset, img_metas)
                
        losses, refine_targets = self.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # get pose refinement loss
        # TODO 虽然传入了深度信息，但是未使用
        losses = self.forward_refine(memory, mlvl_masks, refine_targets,
                                        losses, img_metas)
        return losses

    @force_fp32(apply_to=('all_cls_scores', 'all_kpt_preds'))
    def loss(self,
                all_cls_scores,
                all_kpt_preds,
                all_depths_preds,
                enc_cls_scores,
                enc_kpt_preds,
                enc_depth_preds,
                enc_hm_proto,
                gt_bboxes_list,
                gt_labels_list,
                gt_keypoints_list,
                gt_areas_list,
                dataset,
                img_metas,
                gt_bboxes_ignore=None):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape [nb_dec, bs, num_query, cls_out_channels].
                [3, bs, 300, 1]
            all_kpt_preds (Tensor): Sigmoid regression outputs of all decode layers. 
                Each is a 4D-tensor with normalized coordinate format (x_{i}, y_{i}), 
                and shape [nb_dec, bs, num_query, K*2].
                [3, bs, 300, 2*15]
            all_depths_preds: reference point absolute depth, other keypoints relative depth.
                [3, bs, 300, 1 + 15]
            enc_cls_scores (Tensor): Classification scores of points on encode feature map, 
                has shape (N, sum(h*w), num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
                [bs, sum(h*w), 1]
            enc_kpt_preds (Tensor): Regression results of each points, on the encode 
                feature map, has shape (N, h*w, K*2). Only be passed when as_two_stage is True, \
                otherwise is None.
                [bs, sum(h*w), 2*15]
            enc_depth_preds: reference point absolute depth, other keypoints relative depth.
                Regression results of each points, on the encode  feature map.
                [bs, sum(h*w), 1 + 15]
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
                bs * [num_gts, 4]
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
                bs * [num_gts, ]
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, 15, 11) come from SMAP format.
                bs * [num_gts, 15, 11]
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
                bs * [num_gts, ]
            img_metas (list[dict]): List of image meta information.
                bs * {img info}
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
                None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        num_dec_layers = len(all_cls_scores)  # num decoder = 3, 用来复制gt信息。
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_keypoints_list = [gt_keypoints_list for _ in range(num_dec_layers)]
        all_gt_areas_list = [gt_areas_list for _ in range(num_dec_layers)]
        dataset_list = [dataset for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # 修改输入，修改返回值
        losses_cls, losses_kpt, losses_oks, losses_depth, area_targets_list, \
        kpt_preds_list, depth_preds_list, \
        kpt_weights_list, depth_weights_list, \
        kpt_targets_list,  depth_targets_list = multi_apply(
                self.loss_single, all_cls_scores, all_kpt_preds, all_depths_preds,
                all_gt_labels_list, all_gt_keypoints_list,
                all_gt_areas_list, dataset_list, img_metas_list)  # 计算decoder输出产生的loss

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_losses_cls, enc_losses_kpt, enc_losses_depth = \
                self.loss_single_rpn(
                    enc_cls_scores, enc_kpt_preds, enc_depth_preds, binary_labels_list,
                    gt_keypoints_list, gt_areas_list, dataset, img_metas)
            
            loss_dict['enc_loss_cls'] = enc_losses_cls
            loss_dict['enc_loss_kpt'] = enc_losses_kpt
            loss_dict['enc_loss_depth'] = enc_losses_depth

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_kpt'] = losses_kpt[-1]
        loss_dict['loss_oks'] = losses_oks[-1]
        loss_dict['loss_depth'] = losses_depth[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_kpt_i, loss_oks_i, loss_depth_i in zip(
                losses_cls[:-1], losses_kpt[:-1], losses_oks[:-1], losses_depth[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_kpt'] = loss_kpt_i
            loss_dict[f'd{num_dec_layer}.loss_oks'] = loss_oks_i
            loss_dict[f'd{num_dec_layer}.loss_depth'] = loss_depth_i
            num_dec_layer += 1

        # losses of heatmap generated from P3 feature map
        hm_pred, hm_mask = enc_hm_proto
        loss_hm = self.loss_heatmap(hm_pred, hm_mask, gt_keypoints_list,
                                    gt_labels_list, gt_bboxes_list)
        loss_dict['loss_hm'] = loss_hm

        return loss_dict, (area_targets_list[-1],
            kpt_preds_list[-1], kpt_targets_list[-1], kpt_weights_list[-1],
            depth_preds_list[-1], depth_targets_list[-1], depth_weights_list[-1],)

    def loss_heatmap(self, hm_pred, hm_mask, gt_keypoints, gt_labels,
                        gt_bboxes):
        """
        这里的gt_keypoints与之前不同, 进行修正
        """
        assert hm_pred.shape[-2:] == hm_mask.shape[-2:]
        num_img, _, h, w = hm_pred.size()
        # placeholder of heatmap target (Gaussian distribution)
        hm_target = hm_pred.new_zeros(hm_pred.shape)
        for i, (gt_label, gt_bbox, gt_keypoint_drepth) in enumerate(
                zip(gt_labels, gt_bboxes, gt_keypoints)):
            if gt_label.size(0) == 0:
                continue
            gt_keypoint = torch.cat((gt_keypoint_drepth[..., :2], gt_keypoint_drepth[..., 3].unsqueeze(-1)), -1)  # [x, y, vis]
            gt_keypoint = gt_keypoint.reshape(gt_keypoint.shape[0], -1,
                                                3).clone()
            gt_keypoint[..., :2] /= 8
            if gt_keypoint[..., 0].max() > w or gt_keypoint[..., 1].max() > h:
                import pdb;pdb.set_trace() 
            # FIXME 为什么会有 keypoint 越界
            # gt_keypoint[..., 0] = torch.clip(gt_keypoint[..., 0], 0, w - 1)
            # gt_keypoint[..., 1] = torch.clip(gt_keypoint[..., 1], 0, h - 1)
            # assert gt_keypoint[..., 0].max() <= w  # new coordinate system
            # assert gt_keypoint[..., 1].max() <= h  # new coordinate system
            gt_bbox /= 8
            gt_w = gt_bbox[:, 2] - gt_bbox[:, 0]
            gt_h = gt_bbox[:, 3] - gt_bbox[:, 1]
            for j in range(gt_label.size(0)):
                # get heatmap radius
                kp_radius = torch.clamp(
                    torch.floor(
                        gaussian_radius((gt_h[j], gt_w[j]), min_overlap=0.9)),
                    min=0, max=3)
                for k in range(self.num_keypoints):
                    if gt_keypoint[j, k, 2] > 0:
                        gt_kp = gt_keypoint[j, k, :2]
                        gt_kp_int = torch.floor(gt_kp)
                        draw_umich_gaussian(hm_target[i, k], gt_kp_int,
                                            kp_radius)
        # compute heatmap loss
        hm_pred = torch.clamp(
            hm_pred.sigmoid_(), min=1e-4, max=1 - 1e-4)  # refer to CenterNet
        loss_hm = self.loss_hm(hm_pred, hm_target, mask=~hm_mask.unsqueeze(1))
        return loss_hm

    def loss_single(self,
                    cls_scores,
                    kpt_preds,
                    depth_preds,
                    gt_labels_list,
                    gt_keypoints_list,
                    gt_areas_list,
                    dataset_list,
                    img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i})
                shape [bs, num_query, K*2].
            depth_preds: reference point depth and other keypoints releative depth
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, 15, 11) come from SMAP format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)  # batch_size of imgs
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]  # [bs, 300, 1] -> bs * [300, 1]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        depth_preds_list = [depth_preds[i] for i in range(num_imgs)]

        cls_reg_depth_targets = self.get_targets(cls_scores_list, kpt_preds_list, depth_preds_list,
                gt_labels_list, gt_keypoints_list, gt_areas_list, dataset_list, img_metas)
        
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list, depth_targets_list, 
            depth_weights_list, area_targets_list, num_total_pos, num_total_neg) = cls_reg_depth_targets
        
        # 将所有batch拼接在一起
        labels = torch.cat(labels_list, 0)  # [bs * 300, ]
        label_weights = torch.cat(label_weights_list, 0)  # [bs * 300, ]
        kpt_targets = torch.cat(kpt_targets_list, 0)  # [bs * 300, 30]
        kpt_weights = torch.cat(kpt_weights_list, 0)  # [bs * 300, 30]
        depth_targets = torch.cat(depth_targets_list, 0)  # [bs * 300, 16]
        depth_weights = torch.cat(depth_weights_list, 0)  # [bs * 300, 16]
        area_targets = torch.cat(area_targets_list, 0)  # [bs * 300, ]

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale keypoints
        factors = []
        for img_meta, kpt_pred in zip(img_metas, kpt_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = kpt_pred.new_tensor([img_w, img_h, 
                        img_w, img_h]).unsqueeze(0).repeat(kpt_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)  # [bs*300, 4]

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])  # [bs * 300, 30]
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_kpt(
            kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)
        
        # keypoint oks loss
        pos_inds = kpt_weights.sum(-1) > 0
        factors = factors[pos_inds][:, :2].repeat(1, kpt_preds.shape[-1] // 2)
        pos_kpt_preds = kpt_preds[pos_inds] * factors
        pos_kpt_targets = kpt_targets[pos_inds] * factors
        pos_areas = area_targets[pos_inds]
        pos_valid = kpt_weights[pos_inds, 0::2]
        if len(pos_areas) == 0:
            loss_oks = pos_kpt_preds.sum() * 0
        else:
            # if not (pos_areas > 0).all():
            #     import pdb;pdb.set_trace()
            assert (pos_areas > 0).all(), f"img_meta{img_meta}"
            loss_oks = self.loss_oks(
                pos_kpt_preds,
                pos_kpt_targets,
                pos_valid,
                pos_areas,
                avg_factor=num_total_pos)
        # depth L1 Loss 
        # 使用depth_weights和img_metas['dataset']在signle_target去控制是否计算深度loss, 故这里不在判断数据集类型
        depth_preds = depth_preds.reshape(-1, depth_preds.shape[-1])  # [bs * 300, 1 + 15]
        num_valid_depth = torch.clamp(
            reduce_mean(depth_weights.sum()), min=1).item()
        # 处理关键点的相对深度 至 绝对深度
        # TODO 为什么不支持 [..., 1:] += [..., 0] 
        refer_center_depth = depth_preds[..., 0]  # [bs * 300, ]
        refer_center_depth_tmp = refer_center_depth.unsqueeze(-1).repeat(1, 15)  # [bs * 300, 15]
        kpt_real_depth = depth_preds[..., 1:]  # [bs * 300, 15]
        kpt_real_depth_tmp = kpt_real_depth + refer_center_depth_tmp  # [bs * 300, 15]
        kpt_depth_preds = torch.cat((refer_center_depth.unsqueeze(-1), kpt_real_depth_tmp), -1)
        loss_depth = self.loss_depth(
            kpt_depth_preds, depth_targets, depth_weights, avg_factor=num_valid_depth)
        
        return loss_cls, loss_kpt, loss_oks, loss_depth, area_targets,\
            kpt_preds, depth_preds, \
            kpt_weights, depth_weights, \
            kpt_targets, depth_targets

    def get_targets(self,
                    cls_scores_list,
                    kpt_preds_list,
                    depth_preds_list,
                    gt_labels_list,
                    gt_keypoints_list,
                    gt_areas_list,
                    dataset_list,
                    img_metas):
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels]. len=batch_size
            kpt_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (x_{i}, y_{i}) and shape [num_query, K*2]. len=batch_size
            depth_preds_list:
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3).
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                length = batch_size
                - labels_list (list[Tensor]): Labels for all images.  # [300, ]
                - label_weights_list (list[Tensor]): Label weights for all
                    images.  # [300, ]
                - kpt_targets_list (list[Tensor]): Keypoint targets for all
                    images.  # [300, 15*2 + 1 + 15]
                - kpt_weights_list (list[Tensor]): Keypoint weights for all
                    images.  # [300, 15*2 + 1 + 15]
                - area_targets_list (list[Tensor]): area targets for all
                    images.  # [300. ]
                - num_total_pos (int): Number of positive samples in all
                    images.  # eg: [279, 296], num = 2 
                - num_total_neg (int): Number of negative samples in all
                    images.  # eg: [0, ..., 299], num = 298
        """
        # len(labels_list) == batch_size
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list, depth_targets_list, 
            depth_weights_list, area_targets_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, kpt_preds_list, depth_preds_list,
                gt_labels_list, gt_keypoints_list, gt_areas_list, dataset_list, img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))  # pos 个数
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))  # neg 个数
        return (labels_list, label_weights_list, kpt_targets_list,
                kpt_weights_list, depth_targets_list, depth_weights_list, 
                area_targets_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                            cls_score,
                            kpt_pred,
                            depth_pred,
                            gt_labels,
                            gt_keypoints,
                            gt_areas,
                            dataset,
                            img_meta):
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
                [300, 1]
            kpt_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (x_{i}, y_{i}) and
                shape [num_query, K*2].
                [300, 30]
            depth_preds: 
                [300, 16]
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
                [num_gts, ]
            gt_keypoints (Tensor): Ground truth keypoints for one image with
                shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v] format.
                [num_gts, 15, 11]
            gt_areas (Tensor): Ground truth mask areas for one image
                with shape (num_gts, ).
                [num_gts, ]
            img_meta (dict): Meta information for one image.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor): Label weights of each image.
                - kpt_targets (Tensor): Keypoint targets of each image.
                - kpt_weights (Tensor): Keypoint weights of each image.
                - depth_targets (Tensor): Depth targets of each image.
                - depth_weights (Tensor): Depth weights of each image.
                - area_targets (Tensor): Area targets of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_querys = kpt_pred.size(0)
        # # 将ground truth 转为 float32
        # gt_keypoints = gt_keypoints.to(torch.float32)
        # gt_areas = gt_areas.to(torch.float32)
        # assigner and sampler
        assign_result = self.assigner.assign(cls_score, kpt_pred, depth_pred, 
                            gt_labels, gt_keypoints, gt_areas, dataset, img_meta)
        sampling_result = self.sampler.sample(assign_result, kpt_pred,
                                        gt_keypoints)
        pos_inds = sampling_result.pos_inds  # query中选中的index eg: [279, 296]
        neg_inds = sampling_result.neg_inds  # query中未选中的index eg: [0, ..., 299]

        # label targets
        labels = gt_labels.new_full((num_querys, ),
                                    self.num_classes,
                                    dtype=torch.long)  # [300, ] value= 1 = self.num_classes
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        # sampling_result.pos_assigned_gt_inds: gt 中对应的索引，eg [1, 0]
        # sampling_result.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        label_weights = gt_labels.new_ones(num_querys)  # [300, ]， 这里是全1

        img_h, img_w, _ = img_meta['img_shape']
        # keypoint targets
        kpt_targets = torch.zeros_like(kpt_pred)  # [300, 15*2]
        kpt_weights = torch.zeros_like(kpt_pred)  # [300, 15*2]
        pos_gt_kpts = gt_keypoints[sampling_result.pos_assigned_gt_inds]  # [num_gts, 15, 11]
        try:
            valid_idx = pos_gt_kpts[..., 3] > 0  # vis, [num_gts, 15]
        except:
            print("error")
            import pdb;pdb.set_trace()
        pos_kpt_weights = kpt_weights[pos_inds].reshape(
            pos_gt_kpts.shape[0], kpt_weights.shape[-1] // 2, 2)  # [num_gts, 15, 2]
        pos_kpt_weights[valid_idx] = 1.0
        kpt_weights[pos_inds] = pos_kpt_weights.reshape(
            pos_kpt_weights.shape[0], kpt_pred.shape[-1])  # [300, 34]

        factor = kpt_pred.new_tensor([img_w, img_h]).unsqueeze(0)  # [1, 2]
        pos_gt_kpts_normalized = pos_gt_kpts[..., :2]  # [num_gts, 15, 2]
        pos_gt_kpts_normalized[..., 0] = pos_gt_kpts_normalized[..., 0] / \
            factor[:, 0:1]  # [num_gts, 15, 2]
        pos_gt_kpts_normalized[..., 1] = pos_gt_kpts_normalized[..., 1] / \
            factor[:, 1:2]  # [num_gts, 15, 2]
        kpt_targets[pos_inds] = pos_gt_kpts_normalized.reshape(
            pos_gt_kpts.shape[0], kpt_pred.shape[-1])  # [num_gts, 2]

        # FIXME depth target
        if dataset == "COCO":
            depth_targets = torch.zeros_like(depth_pred)
            depth_weights = torch.zeros_like(depth_pred)
        elif dataset == "MUCO":
            depth_targets = torch.zeros_like(depth_pred)  # [300, 1 + 15]
            depth_weights = torch.zeros_like(depth_pred)  # [300, 1 + 15]
            gt_keypoints_tmp = gt_keypoints[sampling_result.pos_assigned_gt_inds]
            pos_gt_kpts = gt_keypoints_tmp  # [num_gts, 15, 11]
            valid_idx = pos_gt_kpts[:, :, 3] > 0  # vis, [num_gts, 15]
            # depth_weights[pos_inds][..., 0] = 1.0  # reference point 的 weight 直接设为1
            # depth_weights[pos_inds][..., 1:] = valid_idx.int()  # 形状要对应上
            # TODO 为什么无法赋值
            num_gts = gt_keypoints.shape[0]
            refer_point_weights = torch.ones((num_gts, 1)).cuda()  # [num_gts, 1]
            depth_weights[pos_inds] = torch.cat((refer_point_weights,valid_idx.int()), -1)  # [num_gts, 1 + 15]
            # 这里直接对gt_targets进行变换，之后计算loss时就不用再针对进行变换
            # FIXME, 是否除数为零
            kpt_gt_depth = gt_keypoints_tmp[..., 6] * img_w / (gt_keypoints_tmp[..., 7] + 1)  # [num_gts, 15]
            kpt_center_depth = torch.sum(kpt_gt_depth, -1) / torch.sum(valid_idx.int(), -1)  # [num_gts, ]
            depth_targets[pos_inds] = torch.cat((kpt_center_depth.unsqueeze(-1), kpt_gt_depth), -1)
            if torch.isnan(depth_targets).int().sum():
                import pdb;pdb.set_trace()
        else:
            raise  NotImplementedError("未知的dataset in _get_target_single.")

        # area target
        area_targets = kpt_pred.new_zeros(
            kpt_pred.shape[0])  # get areas for calculating oks
        pos_gt_areas = gt_areas[sampling_result.pos_assigned_gt_inds]
        area_targets[pos_inds] = pos_gt_areas

        return (labels, label_weights, kpt_targets, kpt_weights, depth_targets, depth_weights,
            area_targets, pos_inds, neg_inds)

    def loss_single_rpn(self,
                        cls_scores,
                        kpt_preds,
                        depth_preds,
                        gt_labels_list,
                        gt_keypoints_list,
                        gt_areas_list,
                        dataset,
                        img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        depth_preds_list = [depth_preds[i] for i in range(num_imgs)]
        cls_reg_depth_targets = self.get_targets(cls_scores_list, kpt_preds_list, depth_preds_list,
                gt_labels_list, gt_keypoints_list, gt_areas_list, dataset, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list, depth_targets_list,
            depth_weights_list, area_targets_list, num_total_pos, num_total_neg) = cls_reg_depth_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        kpt_targets = torch.cat(kpt_targets_list, 0)
        kpt_weights = torch.cat(kpt_weights_list, 0)
        depth_targets = torch.cat(depth_targets_list)
        depth_weights = torch.cat(depth_weights_list)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        # num_total_pos = loss_cls.new_tensor([num_total_pos])
        # num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_kpt_rpn(
            kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)

        # keypoint Depth L1 loss
        depth_preds = depth_preds.reshape(-1, depth_preds.shape[-1])
        assert depth_preds.shape[-1] == 16, f"形状和设想不一样。"
        depth_preds[..., 1:] = depth_preds[..., 1:] + depth_preds[..., 0].unsqueeze(-1)
        num_valid_depth = torch.clamp(
            reduce_mean(depth_weights.sum()), min=1).item()
        loss_depth = self.loss_depth_rpn(
            depth_preds, depth_targets, depth_weights, avg_factor=num_valid_depth)
        return loss_cls, loss_kpt, loss_depth

    @force_fp32(apply_to=('all_cls_scores', 'all_kpt_preds'))
    def get_bboxes(self,
                    all_cls_scores,
                    all_kpt_preds,
                    enc_cls_scores,
                    enc_kpt_preds,
                    hm_proto,
                    memory,
                    mlvl_masks,
                    img_metas,
                    rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of points on
                encode feature map, has shape (N, h*w, num_classes).
                Only be passed when as_two_stage is True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Defalut False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 3-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box. The third item is an (n, K, 3) tensor
                with [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        cls_scores = all_cls_scores[-1]  # [bs, 300, 1]
        kpt_preds = all_kpt_preds[-1]  # [bx, 300, 34]
        # cls_scores = enc_cls_scores
        # kpt_preds = enc_kpt_preds

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            kpt_pred = kpt_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # TODO: only support single image test
            # memory_i = memory[:, img_id, :]
            # mlvl_mask = mlvl_masks[img_id]
            proposals = self._get_bboxes_single(cls_score, kpt_pred,
                                                img_shape, scale_factor,
                                                memory, mlvl_masks, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                            cls_score,
                            kpt_pred,
                            img_shape,
                            scale_factor,
                            memory,
                            mlvl_masks,
                            rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (x_{i}, y_{i}) and
                shape [num_query, K*2].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5],
                    where the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with
                    shape [num_query].
                - det_kpts: Predicted keypoints with shape [num_query, K, 3].
        """
        assert len(cls_score) == len(kpt_pred)  # 300
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)  # 100
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()  # (300, 1)
            scores, indexs = cls_score.view(-1).topk(max_per_img)  # 100
            det_labels = indexs % self.num_classes  # ???
            bbox_index = indexs // self.num_classes  # 100
            kpt_pred = kpt_pred[bbox_index]  # (100, 34)
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            kpt_pred = kpt_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        # ----- results after pose decoder -----
        # det_kpts = kpt_pred.reshape(kpt_pred.size(0), -1, 2)

        # ----- results after joint decoder (default) -----
        # import time
        # start = time.time()
        refine_targets = (kpt_pred, None, None, torch.ones_like(kpt_pred))
        refine_outputs = self.forward_refine(memory, mlvl_masks,
                                                refine_targets, None, None)
        # end = time.time()
        # print(f'refine time: {end - start:.6f}')
        det_kpts = refine_outputs[-1]  # [100, 17, 2]
        # img_shape: (1241, 800, 3)
        det_kpts[..., 0] = det_kpts[..., 0] * img_shape[1]  # [100, 17]
        det_kpts[..., 1] = det_kpts[..., 1] * img_shape[0]  # [100, 17]
        det_kpts[..., 0].clamp_(min=0, max=img_shape[1])
        det_kpts[..., 1].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_kpts /= det_kpts.new_tensor(
                scale_factor[:2]).unsqueeze(0).unsqueeze(0)  # [100, 17, 2] / [1, 1, 2]

        # use circumscribed rectangle box of keypoints as det bboxes
        x1 = det_kpts[..., 0].min(dim=1, keepdim=True)[0]
        y1 = det_kpts[..., 1].min(dim=1, keepdim=True)[0]
        x2 = det_kpts[..., 0].max(dim=1, keepdim=True)[0]
        y2 = det_kpts[..., 1].max(dim=1, keepdim=True)[0]
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=1)  # [100, 4]
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)  # [100, 5]

        det_kpts = torch.cat(  # [100, 17, 3]
            (det_kpts, det_kpts.new_ones(det_kpts[..., :1].shape)), dim=2)

        return det_bboxes, det_labels, det_kpts
    
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:n
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.  # True

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The third item is ``kpts`` with shape
                (n, K, 3), in [p^{1}_x, p^{1}_y, p^{1}_v, p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list
