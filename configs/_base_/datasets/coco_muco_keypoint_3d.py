# dataset settings, use 3d dataset
dataset_type = 'opera.JointDataset'
# data_root
data_coco_root = '/home/notebook/data/group/wangxiong/smoothformer/hpe_data/data/coco2017/'
data_muco_root = '/home/notebook/data/group/wangxiong/smoothformer/hpe_data/data/MuCo/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='opera.LoadImgFromFile', to_float32=True),
    dict(
        type='opera.LoadAnnosFromFile', 
        with_dataset=True,
        with_bbox=True,
        with_label=True,
        with_keypoints=True,
        with_areas=True,
    ),
    dict(
        type='mmdet.PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    # dict(
    #     type='opera.VisImg',
    #     draw_bbox=True,
    #     draw_keypoints=True,
    #     img_prefix='before_filp_00',
    #     img_path='/data/jupyter/PETR/opera/datasets/smap_utils/'
    # ),
    dict(
        type='opera.AugRandomFlip',
        flip_ratio=1.0,  # 测试Flip
    ),
    # dict(
    #     type='opera.VisImg',
    #     draw_bbox=True,
    #     draw_keypoints=True,
    #     img_prefix='before_Random_00',
    #     img_path='/data/jupyter/PETR/opera/datasets/smap_utils/'
    # ),
    dict(
        type='opera.AugRandomRotate',
        max_rotate_degree=30,
        rotate_prob=1.0,  # 测试Rotate
    ),
    # dict(
    #     type='opera.VisImg',
    #     draw_bbox=True,
    #     draw_keypoints=True,
    #     img_prefix='before_AugMent_00',
    #     img_path='/data/jupyter/PETR/opera/datasets/smap_utils/'
    # ),
    dict(
        type='mmdet.AutoAugment',
        policies=[
            [
                dict(
                    type='opera.AugResize',
                    img_scale=[(400, 1400), (1400, 1400)],
                    multiscale_mode='range',
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type='opera.AugResize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True,
                ),
                dict(
                    type='opera.AugCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type='opera.AugResize',
                    img_scale=[(400, 1400), (1400, 1400)],
                    multiscale_mode='range',
                    override=True,
                    keep_ratio=True,
                )
            ]
        ]
    ),
    # dict(
    #     type='opera.VisImg',
    #     draw_bbox=True,
    #     draw_keypoints=True,
    #     img_prefix='before_Normalize_00',
    #     img_path='/data/jupyter/PETR/opera/datasets/smap_utils/'
    # ),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=1),  # 使用0填充图像边缘
    dict(type='opera.AugConstraint', 
        with_cons_coord=True,
        with_cons_length=True,
    ),
    dict(type='opera.FormatBundle',
            extra_keys=['gt_keypoints', 'gt_areas',]),
    dict(type='mmdet.Collect',
            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas', 'dataset']),
]

# 尚未修改
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
    ])
]


# 仅使用修改格式后的coco数据集进行调试
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=1,
#     train=dict(
#         type=dataset_type,
#         ann_file= '/home/notebook/code/personal/S9043252/wz/dataset_anno/coco_keypoints_train2017.json',
#         img_prefix=data_coco_root,
#         pipeline=train_pipeline
#     ),
#     val=dict(
#         type=dataset_type,
#         ann_file=[],
#         pipeline=test_pipeline
#     ),
#     test=dict(
#         ann_file=[],
#         pipeline=test_pipeline,
#     )
# )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file= '/home/notebook/code/personal/S9043252/wz/dataset_anno/MuCo.json',
        img_prefix=data_muco_root,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=[],
        pipeline=test_pipeline
    ),
    test=dict(
        ann_file=[],
        pipeline=test_pipeline,
    )
)

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=1,
#     train=dict(
#         type=dataset_type,
#         ann_file=['/home/notebook/code/personal/S9043252/wz/dataset_anno//coco_keypoints_train2017.json',
#                     '/home/notebook/code/personal/S9043252/wz/dataset_anno//MuCo.json'],
#         img_prefix=[data_coco_root, data_muco_root],
#         pipeline=train_pipeline
#     ),
#     val=dict(
#         type=dataset_type,
#         ann_file=[],
#         pipeline=test_pipeline
#     ),
#     test=dict(
#         ann_file=[],
#         pipeline=test_pipeline,
#     )
# )

evaluation = dict(interval=1, metric='keypoints')