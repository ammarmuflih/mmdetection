_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        frozen_stages=-1,
        init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=2),
        mask_head=dict(
            num_classes=2))
    )

dataset_type = 'COCODataset'
classes = ('person','car')
data = dict(
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

evaluation = dict(interval=10)