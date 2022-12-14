_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
cudnn_benchmark = True
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
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
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),

    neck=dict(
        in_channels=[40, 112, 320],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    
#    training and testing settings
#    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)),

    # rpn_head=dict(
    #     type='RPNHead',
    #     in_channels=256,
    #     feat_channels=256,
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         scales=[8],
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[4, 8, 16, 32, 64]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0]),
    #     loss_cls=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(
        bbox_head=dict(
            num_classes=2),
        mask_head=dict(
            num_classes=2))
    
    # train_cfg=dict(
    #     rpn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.7,
    #             neg_iou_thr=0.5,
    #             min_pos_iou=0.3,
    #             match_low_quality=True,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=256,
    #             pos_fraction=0.5,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=False),
    #         allowed_border=-1,
    #         pos_weight=-1,
    #         debug=False),
    #     rpn_proposal=dict(
    #         nms_pre=2000,
    #         max_per_img=1000,
    #         nms=dict(type='nms', iou_threshold=0.7),
    #         min_bbox_size=0),
    #     rcnn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.5,
    #             neg_iou_thr=0.5,
    #             min_pos_iou=0.5,
    #             match_low_quality=True,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=True),
    #         mask_size=28,
    #         pos_weight=-1,
    #         debug=False)),

    # test_cfg=dict(
    #     rpn=dict(
    #         nms_pre=1000,
    #         max_per_img=1000,
    #         nms=dict(type='nms', iou_threshold=0.7),
    #         min_bbox_size=0),
    #     rcnn=dict(
    #         score_thr=0.05,
    #         nms=dict(type='nms', iou_threshold=0.5),
    #         max_per_img=100,
    #         mask_thr_binary=0.5))

)

# # dataset settings
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_size = (896, 896)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='Resize',
#         img_scale=img_size,
#         ratio_range=(0.8, 1.2),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=img_size),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=img_size),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_size,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size=img_size),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

#Perdataset an
dataset_type = 'COCODataset'
classes = ('person','car')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    # train=dict(pipeline=train_pipeline),
    # val=dict(pipeline=test_pipeline),
    # test=dict(pipeline=test_pipeline)
    train=dict(
        img_prefix='/home/ammar/Documents/mmdetection/data/coco/train2017/',
        classes=classes,
        ann_file='/home/ammar/Documents/mmdetection/data/coco/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='/home/ammar/Documents/mmdetection/data/coco/val2017',
        classes=classes,
        ann_file='/home/ammar/Documents/mmdetection/data/coco/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='/home/ammar/Documents/mmdetection/data/coco/train2017/',
        classes=classes,
        ann_file='/home/ammar/Documents/mmdetection/data/coco/annotations/instances_train2017.json')
        
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


