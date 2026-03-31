from mmcv_utils.utils.config import Config
from mmseg_utils.datasets.builder import build_dataloader, build_dataset
from mmseg_utils.datasets.builder import DATASETS

def build_seg_dataset(config, scales=None, split="support"):

    """Build a dataset from config."""
    cfg = Config.fromfile(config)
    cfg_split = getattr(cfg.data, split)
    
    sizes = []
    for size in scales:
        sizes.append((size[0], size[1]))

    if split == "support":
        cfg_split['pipeline'][3]['img_scale'] = sizes
    elif split == "test":
        cfg_split['pipeline'][2]['img_scale'] = sizes

    if sizes[0][0] == 'None':
        
        if split == "support":
            cfg_split['pipeline'][3]['transforms'] = cfg_split['pipeline'][3]['transforms'][1:]
        elif split == "test":
            cfg_split['pipeline'][2]['transforms'] = cfg_split['pipeline'][2]['transforms'][1:]

    dataset = build_dataset(cfg_split)

    return dataset

def build_seg_dataloader(dataset, dist=True):
    # batch size is set to 1 to handle varying image size (due to different aspect ratio)
    if dist:
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=dist,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
        )
    else:
        data_loader = build_dataloader(
            dataset=dataset,
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=dist,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
        )

    return data_loader

