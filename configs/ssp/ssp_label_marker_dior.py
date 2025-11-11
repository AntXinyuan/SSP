_base_ = [
    './ssp_label_marker.py'
]

store_dir = 'work_dirs/ssp_label_marker_dior/'

angle_version = 'le90'
classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
           'bridge', 'chimney', 'expressway-service-area',
           'expressway-toll-station', 'dam', 'golffield',
           'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
           'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
           'windmill')

# model settings
model = dict(
    bbox_head=dict(
        num_classes=len(classes),
        cls_square=[2, 5, 9, 14, 15, 19],
        cls_overlap=[[0, 1], [13, 11], [7, 6]],
        cls_merge=[[0, 5, 7, 13, 14, 15, 17, 18, 19], [1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 16,]],
        cls_stable=None,#[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
        sp_thres=dict(
            default=[0.999, 0.005],
            #efault=[0.995, 0.005],
            #override=(([1, 2, 3, 16], [0.96, 0.005]),),
            #override=(([7, 8, 19], [0.995, 0.005]),),
            #override=(([0, 1, 3, 7, 8, 10, 14], [0.995, 0.005]),),
            confidence=(0.05, 0.6, 0.95))),
    train_cfg=dict(
        store_dir=store_dir,))

dataset_type = 'DIORDataset'
data_root = 'datasets/dior/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Main/trainval.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_subdir=data_root + 'JPEGImages-trainval/',
        img_prefix=data_root + 'JPEGImages-trainval/',
        pipeline=train_pipeline,
        version=angle_version,),)

runner = dict(type='EpochBasedRunner', max_epochs=12)

lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])

# this config is for 4 gpus, where total_bs=4*4=16, total_lr=0.01, total_iter_per_batch=800