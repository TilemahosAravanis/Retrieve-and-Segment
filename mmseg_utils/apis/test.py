# Copyright (c) OpenMMLab. All rights reserved.
import torch
from tqdm import tqdm

import torch.nn.functional as F
from mmseg_utils.ops.wrappers import resize

from core.rns import propose_masks

def slide_inference(model, img, img_meta, rescale):

    img_divisible = model.make_input_divisible(img)

    h_stride, w_stride = model.test_crop_stride, model.test_crop_stride
    h_crop, w_crop = model.test_crop_size, model.test_crop_size
    batch_size, _, h_img_divisible, w_img_divisible = img_divisible.size()
    num_classes = model.num_classes
    h_grids = max(h_img_divisible - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img_divisible - w_crop + w_stride - 1, 0) // w_stride + 1

    img_feats = img_divisible.new_zeros((batch_size, model.vlm_feat_dim, 
                                         h_img_divisible // model.vit_patch_size, w_img_divisible // model.vit_patch_size))
    count_mat = img_divisible.new_zeros((batch_size, 1, 
                                         h_img_divisible // model.vit_patch_size, w_img_divisible // model.vit_patch_size))

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img_divisible)
            x2 = min(x1 + w_crop, w_img_divisible)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img_divisible[:, :, y1:y2, x1:x2]
            
            crop_seg_feats = model.forward_pass(crop_img) # extract feats for crop
            crop_seg_feats = crop_seg_feats / crop_seg_feats.norm(dim=1, keepdim=True) # norm channel dim

            temp = F.pad(crop_seg_feats,
                        (int(x1 // model.vit_patch_size), 
                         int(img_feats.shape[3] - (x2 // model.vit_patch_size)), 
                         int(y1 // model.vit_patch_size),
                         int(img_feats.shape[2] - (y2 // model.vit_patch_size))))
             
            img_feats += temp

            count_mat[:, :, y1 // model.vit_patch_size : y2 // model.vit_patch_size,
                        x1 // model.vit_patch_size : x2 // model.vit_patch_size] += 1
            
    assert (count_mat == 0).sum() == 0
 
    img_feats = img_feats / count_mat
    img_feats = img_feats / img_feats.norm(dim=1, keepdim=True) # norm channel dim
    
    B, C, hf, wf = img_feats.shape

    if model.mask_proposal != 'None':
        img_feats, proposed_masks = propose_masks(model, img, img_feats)

    output = model(img_feats)

    if model.mask_proposal != 'None':
        
        # map the output back to the original image size based on the proposed masks
        output = output[proposed_masks]
        output = output.permute(2, 0, 1).unsqueeze(0) # (1, num_classes, H, W)
        
    else:
        #### FINAL OUTPUT ####
        output = output.unsqueeze(0).permute(0, 2, 1).reshape(1, model.num_classes, hf, wf)
    
        output = resize(
            input=output,
            size=(hf * model.vit_patch_size, wf * model.vit_patch_size),
            mode='bilinear',
            align_corners=False)
        
        # crop to original img size
        output = output[:, :, :img.shape[-2], :img.shape[-1]]

    preds = output

    if rescale:
        # remove padding area
        resize_shape = img_meta.data[0][0]['img_shape'][:2]
        preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
        preds = resize(
            preds,
            size=img_meta.data[0][0]['ori_shape'][:2],
            mode='bilinear',
            align_corners=False,
            warning=False)
    
    return preds


def inference(model, img, img_meta, rescale):

    seg_logit = slide_inference(model, img, img_meta, rescale)
    output = F.softmax(seg_logit, dim=1)

    return output

def simple_test(model, img, img_meta, rescale=True):
    """Simple test with single image."""
    
    img = img.cuda()
    
    seg_logit = inference(model, img, img_meta, rescale)
    seg_pred = seg_logit.argmax(dim=1)
    if torch.onnx.is_in_onnx_export():
        # our inference backend only support 4D output
        seg_pred = seg_pred.unsqueeze(0)
        return seg_pred
    seg_pred = seg_pred.cpu().numpy()
    # unravel batch dim
    seg_pred = list(seg_pred)
    return seg_pred


def forward_test(model, img, img_metas, **kwargs):
    # all images in the same aug batch all of the same ori_shape and pad
    # shape
    
    return simple_test(model, img[0], img_metas[0], **kwargs)


def forward(model, img, img_metas, **kwargs):
    """Calls either :func:`forward_train` or :func:`forward_test` depending
    on whether ``return_loss`` is ``True``.

    Note this setting will change the expected inputs. When
    ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
    and List[dict]), and when ``resturn_loss=False``, img and img_meta
    should be double nested (i.e.  List[Tensor], List[List[dict]]), with
    the outer list indicating test time augmentations.
    """
    
    return forward_test(model, img, img_metas, **kwargs)


def model_validation(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset    
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in tqdm(zip(loader_indices, data_loader)):
        
        data.pop("gt_semantic_seg", None)

        with torch.no_grad():
            result = forward(model=model, **data)

        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)

    return results