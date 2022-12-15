_base_ = './mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
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
    samples_per_gpu=1,
    workers_per_gpu=1
    # train=dict(pipeline=train_pipeline),
    # val=dict(pipeline=test_pipeline),
    # test=dict(pipeline=test_pipeline)
)

runner = dict(type='EpochBasedRunner', max_epochs=100)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=1)

evaluation = dict(interval=10)

