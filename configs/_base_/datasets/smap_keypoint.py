# dataset settings
dataset_type = 'opera.JointDataset'
# data_root
data_coco_root = '/data/coco/'
data_muco_root = '/data/MuCo/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train pipeline
train_pipeline = [
    dict(type='opera.LoadImageFromFile', to_float32=True),
    dict(
        type='opera.LoadSmapAnnotations', 
        with_dataset=True,
        with_bbox=True,
        with_keypoints=True,
    ),
    dict(
        type='mmdet.PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(
        type='opera.AugRandomFlip',
        flip_ratio=0.5,
    ),
    dict(
        type='opera.AugRandomRotate',
        max_rotate_degree=30,
        rotate_prob=0.5,
    ),
    dict(
        type='mmdet.AutoAugment',
        policies=[
            [
                dict(
                    type='opera.AugResize',
                    img_scale=[(400, 1400), (1400, 1400)],
                    multiscale_mode='range',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='opera.AugResize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='opera.AugCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='opera.AugResize',
                    img_scale=[(400, 1400), (1400, 1400)],
                    multiscale_mode='range',
                    override=True,
                    keep_ratio=True)
            ]
        ]
    ),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=1),
    dict(type='opera.DefaultFormatBundle',
            extra_keys=['gt_keypoints', 'gt_areas']),
    dict(type='mmdet.Collect',
            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas']),
]

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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=[data_coco_root + 'annotations/coco_keypoints_train2017.json',
                    data_muco_root + 'annotations/MuCo.json'],
        img_prefix=[data_coco_root, data_muco_root],
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

evaluation = dict(interval=1, metric='keypoints')