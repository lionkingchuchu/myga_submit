# dataset settings
dataset_type = 'CocoDataset'
data_root = '/input/jungtdet/data_coco/'
classes = ('oil tank')

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        metainfo = dict(classes=classes),
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train_label.json',
        data_prefix=dict(img='train/img/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo = dict(classes=classes),
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train_label.json',
        data_prefix=dict(img='train/img/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'train/train_label.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
     batch_size=1,
     num_workers=2,
     persistent_workers=True,
     drop_last=False,
     sampler=dict(type='DefaultSampler', shuffle=False),
     dataset=dict(
         metainfo = dict(classes=classes),
         type=dataset_type,
         data_root=data_root,
         ann_file=data_root + 'val/valtest.json',
         data_prefix=dict(img='val/img/'),
         test_mode=True,
         pipeline=test_pipeline))
test_evaluator = dict(
     type='CocoMetric',
     metric='bbox',
     format_only=True,
     ann_file=data_root + 'val/valtest.json',
     outfile_prefix='/output')