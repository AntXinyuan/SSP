_base_ = [
    './ssp_label_marker.py'
]


runner = dict(type='EpochBasedRunner', max_epochs=12)

lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])

# this config is for 4 gpus, where total_bs=4*4=16, total_lr=0.01, total_iter_per_batch=800