_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

import mmdet
mmdet.datasets.coco.CocoDataset.CLASSES=('person','car')
evaluation = dict(interval=10)

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None))
    # neck=dict(
    #     in_channels=[48, 136, 384]))
    # training and testing settings
    #train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

