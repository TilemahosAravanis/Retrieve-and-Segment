import os
import argparse
import gc
import sys
import datasets.transforms

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from hydra import compose, initialize

from tqdm import tqdm
from helpers.visualization import mask2rgb

from models import build_model
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset

from mmseg_utils.apis.test import model_validation


def make_input_divisible(self, x: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """Pad some pixels to make the input size divisible by the patch size."""
    B, _, H_0, W_0 = x.shape
    pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
    pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size
    x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=pad_value)
    
    return x

def slide_feat_extract(model, img, crop_size, stride):

    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.size()
    num_classes = model.num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    
    img_feats = img.new_zeros((batch_size, model.vlm_feat_dim, h_img // model.vit_patch_size, w_img // model.vit_patch_size))
    count_mat = img.new_zeros((batch_size, 1, h_img // model.vit_patch_size, w_img // model.vit_patch_size))

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]

            crop_seg_feats = model.get_vlm_features(crop_img)[0] # (B, C, H, W) 

            crop_seg_feats = crop_seg_feats / crop_seg_feats.norm(dim=1, keepdim=True)
            temp = nn.functional.pad(crop_seg_feats,
                        (int(x1 // model.vit_patch_size), int(img_feats.shape[3] - (x2 // model.vit_patch_size)), int(y1 // model.vit_patch_size),
                        int(img_feats.shape[2] - (y2 // model.vit_patch_size))))
             
            img_feats += temp

            count_mat[:, :, y1 // model.vit_patch_size : y2 // model.vit_patch_size,
                        x1 // model.vit_patch_size : x2 // model.vit_patch_size] += 1

    assert (count_mat == 0).sum() == 0
   
    img_feats = img_feats / count_mat
    
    return img_feats

def gt_mask_pooling(model, img, ann, img_name, num_classes, feats, patch_size, dim_patches_h, dim_patches_w, reduce_zero_label):
        
    # Downsample the masks to patch_size x patch_size
    ann = torch.from_numpy(ann).squeeze(0).cuda()
    ann_downsampled = ann.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    ann_downsampled = ann_downsampled.reshape(dim_patches_h*dim_patches_w, -1)

    if reduce_zero_label == True:
        ann_downsampled = ann_downsampled - 1

    # Average on top of the ground truth masks
    class_indices = torch.arange(0, num_classes, dtype=torch.uint8)
    masks = torch.cat([((ann_downsampled.unsqueeze(0) == i).sum(dim=-1, keepdim=False) / (ann_downsampled.unsqueeze(0) == i).sum().item())
                            for i in class_indices], dim=0).float() # (num_classes, H*W)

    # get an average of the features using the masks
    instance_embeddings = torch.matmul(feats, masks.T).squeeze() # (C, num_classes)
    
    return instance_embeddings.T


@torch.no_grad()
def create_support_set(cfg, dataset_key, model):

    # Initialize the visual support dataset (train set of the respective benchmark)
    supp_loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(dataset_key), scales=cfg.support.scales, split="support"), dist=False)

    # part of the train set since we are interested in few shots
    if len(supp_loader.dataset) > cfg.support.max_samples:
        
        np.random.seed(cfg.support.support_seed)
        indices = np.random.choice(len(supp_loader.dataset), cfg.support.max_samples, replace=False)
        
    else:
        indices = np.arange(len(supp_loader.dataset))
    
    supp_loader.dataset.img_infos = [supp_loader.dataset.img_infos[i] for i in indices]
    model.support_images = supp_loader.dataset
    model.reduce_zero_label = supp_loader.dataset.reduce_zero_label
    
    model.experiment = cfg.experiment   
    model.dataset_key = dataset_key
    
    support = cfg.support
    test = cfg.test
    method = cfg.method
    model_cfg = cfg.model

    model.fewshot = support.images_per_class
    model.seed = support.support_seed
    model.k = support.k
    model.drop_classes_fraction = support.drop_classes_fraction
    model.drop_text_fraction = support.drop_text_fraction
    model.num_classes = len(supp_loader.dataset.CLASSES)
    model.patch_size = model_cfg.vit_patch_size
    model.vlm_feat_dim = support.vlm_feat_dim
    model.crop_sizes = support.crop_sizes
    model.strides = support.strides

    model.test_crop_size = test.test_crop_size
    model.test_crop_stride = test.test_crop_stride

    model.mask_proposal = method.mask_proposal_strategy
    model.use_text = method.use_text
    model.class_score_Temp = method.class_score_temperature
    model.beta_mixed = method.beta_mixed
    model.beta_pseudo = method.beta_pseudo
    model.lr = method.lr
    model.batch_size = method.batch_size
    model.epochs = method.epochs

    class_indices = torch.arange(0, model.num_classes, dtype=torch.int32)
    
    support_labels_img = []
    for i, img_dict in tqdm(enumerate(model.support_images)):
        
        ann = img_dict['gt_semantic_seg'][0]
        ann = torch.from_numpy(ann).cuda().flatten()

        if model.reduce_zero_label == True:
            ann = ann - 1
        
        support_labels_img.append(torch.stack([(ann == i).any(dim=0) for i in class_indices]).unsqueeze(0))
        
    support_labels_img = torch.cat(support_labels_img, dim=0) # (num_mem_images, num_classes)

    # random permutation of the class indices
    generator = torch.Generator().manual_seed(cfg.support.support_seed)
    class_indices = class_indices[torch.randperm(class_indices.size(0), generator=generator)]

    number_of_imgs_per_class = torch.zeros(model.num_classes, dtype=torch.int32).to('cuda')

    mask = torch.zeros(support_labels_img.shape[0], dtype=torch.bool).to('cuda')
    
    for i, idx in enumerate(class_indices):
            
        mask_class = support_labels_img[:, idx]

        # print the number of true indices aka the number of images that contain this class
        print(f"Class {idx}: {mask_class.sum().item()} images")
    
        true_indices = torch.nonzero(mask_class).squeeze(0)
        
        # create a random permutation of the indices
        generator = torch.Generator().manual_seed(cfg.support.support_seed)
    
        # if true_indices is empty, continue
        if true_indices.numel() == 0:
            print("No images for some class")
            sys.exit(1)
        
        # Shuffle the indices
        true_indices = true_indices[torch.randperm(true_indices.size(0), generator=generator)]
        
        num_indices = (cfg.support.images_per_class - number_of_imgs_per_class[idx].item())
        
        if num_indices <= 0:
            continue
       
        true_indices = true_indices[:num_indices]
        
        for index in class_indices:
            number_of_imgs_per_class[index] += support_labels_img[true_indices, index].sum().item()

        result = torch.zeros_like(mask_class, dtype=torch.bool)
        result[true_indices] = True
        
        result = result.to('cuda')
        
        mask = mask | result
    
    # keep only the model.support_images that are in the mask
    model.support_images.img_infos = [model.support_images.img_infos[i] for i in range(len(model.support_images)) if mask[i]]
    model.labels_per_support_image = support_labels_img[mask]
  
    # Iterate over the support images and extract visual features
    for i, img_dict in tqdm(enumerate(model.support_images)):
        
        imgs = img_dict['img']
        anns = img_dict['gt_semantic_seg']
        img_name = img_dict['img_metas'][0].data['img_info']['filename']

        total_vlm_feats = torch.zeros((model.num_classes, model.vlm_feat_dim)).cuda() # (C, 512)
        for j in range(len(imgs)):
            img = imgs[j].unsqueeze(0).cuda()
            ann = anns[j]
            
            ann = torch.from_numpy(ann).unsqueeze(0).unsqueeze(0)
            ann = make_input_divisible(model, ann, pad_value=255).squeeze(0).numpy()
            img = make_input_divisible(model, img, pad_value=0)
            
            dim_patches_h = img.shape[-2] // model.patch_size
            dim_patches_w = img.shape[-1] // model.patch_size
            
            vlm_feats = torch.zeros((1, model.vlm_feat_dim, dim_patches_h, dim_patches_w)).cuda() # (1, C, H/P, W/P)
            for crop_size in model.crop_sizes:
                for stride in model.strides:
                    vlm_feats += slide_feat_extract(model, img, crop_size=(crop_size, crop_size), stride=(stride, stride)) # (1, C, H/P, W/P)
            
            vlm_feats = vlm_feats / (len(model.crop_sizes) * len(model.strides))

            vlm_feats = vlm_feats.reshape(vlm_feats.shape[0], model.vlm_feat_dim, -1) # (B, C, H*W)
            vlm_feats = vlm_feats / vlm_feats.norm(dim=1, keepdim=True)

            ### support vector extraction ###
            vlm_feats = gt_mask_pooling(model, img, ann, img_name, len(supp_loader.dataset.CLASSES), 
                                        vlm_feats, model.patch_size, dim_patches_h, dim_patches_w, 
                                        model.reduce_zero_label)

            vlm_feats += vlm_feats / vlm_feats.norm(dim=-1, keepdim=True)
            
            vlm_feats[vlm_feats != vlm_feats] = float('+inf')
            
            total_vlm_feats += vlm_feats
                
        vlm_feats = total_vlm_feats / len(imgs)
        vlm_feats = vlm_feats / vlm_feats.norm(dim=1, keepdim=True) # (num_classes, C) 
        
        # Replace NaNs with 'inf's
        vlm_feats[vlm_feats != vlm_feats] = float('+inf')

        if i == 0:
            support_vlm_feats = [vlm_feats]
        else:
            support_vlm_feats.append(vlm_feats)
    
    support_vlm_feats = torch.cat(support_vlm_feats, dim=0)
    
    print(f"support bank with per-image visual class feats, shape: {support_vlm_feats.shape}")
    
    model.support_vlm_feats = support_vlm_feats

    # randomly select drop number of classes
    drop_classes_num = int(model.drop_classes_fraction * model.num_classes)
    # create a random permutation of the class indices with self.seed and keep the first drop_classes_num indices (use generator)
    generator = torch.Generator().manual_seed(model.seed)
    model.drop_classes = torch.randperm(model.num_classes, generator=generator)[:drop_classes_num]
    model.drop_classes = model.drop_classes.to(model.device)

    model.support_vlm_feats = model.support_vlm_feats.reshape(-1, model.num_classes, model.vlm_feat_dim)
    model.support_vlm_feats[:, model.drop_classes, :] = float('+inf')  
    model.support_vlm_feats = model.support_vlm_feats.reshape(-1, model.vlm_feat_dim)

    # class vectors
    set_class_embeddings(model)
    

def set_class_embeddings(model):
    vlm_feats = model.support_vlm_feats.clone()
    vlm_feats[vlm_feats == float('+inf')] = 0.0
    
    visual = vlm_feats.reshape(-1, model.num_classes, model.vlm_feat_dim).mean(dim=0).cuda()
    visual = visual / visual.norm(dim=1, keepdim=True)
    
    textual = model.text_query_embeddings
    model.num_text_embeddings = textual.shape[0]
    
    textual = textual / textual.norm(dim=1, keepdim=True)

    if model.drop_text_fraction != 0.0:
        drop_text_num = int(model.drop_text_fraction * model.num_classes)
        drop_text_classes = torch.randperm(
            model.num_classes,
            generator=torch.Generator().manual_seed(cfg.support.support_seed)
        )[:drop_text_num]
        drop_text_classes = drop_text_classes.to(model.device)
        
        keep_mask = torch.ones(model.num_classes, dtype=torch.bool, device=model.device)
        keep_mask[drop_text_classes] = False
        rest_mean = textual[keep_mask].mean(dim=0, keepdim=True)
        textual[drop_text_classes] = rest_mean
        
        textual = textual / textual.norm(dim=1, keepdim=True)
    
    mixed = []
    
    if model.use_text:
        lambdas = [0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        
        if textual.shape[0] > model.num_classes:
            class_expansion_len = textual.shape[0] - model.num_classes
            visual = torch.cat([visual[0].unsqueeze(0).expand(class_expansion_len, -1), visual], dim=0)

            mixed_labels = torch.cat([torch.zeros(class_expansion_len, dtype=torch.long).to(model.device),
                                      torch.arange(0, model.num_classes, dtype=torch.long).to(model.device)], dim=0)
        else:
            mixed_labels = torch.arange(0, model.num_classes, dtype=torch.long).to(model.device)    
    
        for lambda_ in lambdas:
            feats = lambda_ * visual + (1 - lambda_) * textual
            feats = feats / feats.norm(dim=1, keepdim=True)
            
            mixed.append(feats)
    else:
        mixed = [torch.empty((0, model.vlm_feat_dim)).to(model.device)]
        mixed_labels = torch.empty((model.num_classes), dtype=torch.long).to(model.device)
        
    mixed = torch.cat(mixed, dim=0)

    model.visual_embeddings = visual
    model.textual_embeddings = textual
    model.mixed = mixed
    model.mixed_labels = mixed_labels
          
def evaluate(cfg, val_loaders):
    ret = {}

    for key, loader in val_loaders.items():

        print(f"### Validation dataset: {key}")
        CLASSES = loader.dataset.CLASSES
        print(f"Creating model:{cfg.model.type}")
        model = build_model(cfg.model, class_names=CLASSES, dataset_key=key)
        
        model.apply_found = False
        model.cuda()
        model.device = "cuda"
        model.eval()
        
        # create visual support set
        create_support_set(cfg, key, model)

        miou, metrics = validate_seg(cfg, cfg.evaluate.get(key), key, loader, model)
        print(f"[{key}] mIoU of {len(loader.dataset)} test images: {miou:.2f}%")
        ret[f"val/{key}_miou"] = miou

    ret["val/avg_miou"] = np.mean([v for k, v in ret.items() if "miou" in k])
    return ret

def validate_seg(config, seg_config, dataset_key, data_loader, model):
    model.eval()

    model.CLASSES = data_loader.dataset.CLASSES
    model.PALETTE = data_loader.dataset.PALETTE

    results = model_validation(
        model=model,
        data_loader=data_loader,
        pre_eval=True,
    )
    
    torch.cuda.empty_cache()
    gc.collect()

    metric = [data_loader.dataset.evaluate(results, metric="mIoU")]
    
    miou_result = metric[0]["mIoU"] * 100
    torch.cuda.empty_cache()
    
    return miou_result, metric

def main(cfg):
    
    cudnn.benchmark = True

    val_loaders = {}
    for key in cfg.evaluate.task:
        loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(key), scales=cfg.test.scales, split="test"), dist=False)
        val_loaders[key] = loader
    
    res = evaluate(cfg, val_loaders)
    
    # Final results
    print(f"\033[91mFinal results:\033[0m")
    for k, v in res.items():
        print(f"\033[91m{k}: {v:.2f}\033[0m")
    print('\n')

def parse_args():
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    parser.add_argument('config', help='config file path')
    args, unknown = parser.parse_known_args()
    import ast

    extra_args = {}
    for u in unknown:
        if '=' in u:
            k, v = u.split('=', 1)
            try:
                extra_args[k] = ast.literal_eval(v)
            except Exception:
                extra_args[k] = v

    return args, unknown

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    args, unknown = parse_args()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config, overrides=unknown)
    main(cfg)