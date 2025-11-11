_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

store_dir = 'work_dirs/ssp_label_marker/'

angle_version = 'le90'
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')

# model settings
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',  # use P5
        num_outs=6,
        relu_before_extra_convs=True),
    # store_dir='rotated_fcos_r50_fpn_1x_dota_le90_2',
    bbox_head=dict(
        type='SSPLabelMarkerHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        regress_ranges=((-1, 32), (32, 64), (64, 128), (128, 256), (256, 512),
                                 (512, 1e8)),
        strides=[4, 8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=False,
        separate_angle=False,
        scale_angle=True,
        cls_square=[1, 9, 11],
        cls_overlap=[[3, 10], [6, 12],],
        cls_merge=[[0, 4, 5, 6, 9, 13, 14], [1, 2, 3, 7, 8, 10, 11, 12]],
        cls_stable=None,#[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
        sp_thres=dict(
            default=[0.999, 0.005],
            override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.995, 0.005])),
            #override=(([0, 1, 3, 7, 8, 10, 14], [0.995, 0.005]),),
            confidence=(0.05, 0.6, 0.95)),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5, #2.0
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.001),
        loss_centerness=dict(type='GaussianFocalLoss', loss_weight=0.0)),
    # training and testing settings
    train_cfg=dict(
        store_dir=store_dir,
        visualize_dir=None,
        pseudo_label_dir=None,),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
#find_unused_parameters = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='RBox2PointWithNoise', p=0.1),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=4,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

runner = dict(type='EpochBasedRunner', max_epochs=6)

lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[4, 5])

evaluation = dict(interval=9999) # do not evaluate during training
optimizer = dict(lr=0.025*4)

custom_hooks = [dict(type='RecordEpochIterHook')]

# this config is for 4 gpus, where total_bs=4*4=16, total_lr=0.01, total_iter_per_batch=800