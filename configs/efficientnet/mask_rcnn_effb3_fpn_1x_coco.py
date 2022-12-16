_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

#Perdataset an
dataset_type = 'COCODataset'
classes = ('person','car')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='/home/ammar/Documents/mmdetection/data/coco/train2017/',
        classes=classes,
        ann_file='/home/ammar/Documents/mmdetection/data/coco/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='/home/ammar/Documents/mmdetection/data/coco/val2017',
        classes=classes,
        ann_file='/home/ammar/Documents/mmdetection/data/coco/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='/home/ammar/Documents/mmdetection/data/coco/test2017/',
        classes=classes,
        ann_file='/home/ammar/Documents/mmdetection/data/coco/annotations/filtered_labels_test.json'))



cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b0',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=None),

    neck=dict(
        type='FPN',
        in_channels=[40, 112, 320],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)),

    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=2),
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=2)
)


# optimizer
optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[8, 11])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=1)

#Eval interval
evaluation = dict(interval=10)


