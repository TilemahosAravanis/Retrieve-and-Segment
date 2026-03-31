_base_ = ["../custom_import.py"]
dataset_type = "ADE20KDataset"
data_root = "./data"

support_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='ToRGB'),
    dict(type='LoadAnnotations'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="ImageToTensorV2", keys=["img"]),
            dict(type="Collect", keys=["img", "gt_semantic_seg"], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'flip', 'img_info']),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='ToRGB'),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="ImageToTensorV2", keys=["img"]),
            dict(type="Collect", keys=["img"], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'flip', 'img_info']),
        ],
    ),
]

data = dict(
    support=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="ADEChallengeData2016/images/training",
        ann_dir="ADEChallengeData2016/annotations/training",
        pipeline=support_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="ADEChallengeData2016/images/validation",
        ann_dir="ADEChallengeData2016/annotations/validation",
        pipeline=test_pipeline,
    )
)

