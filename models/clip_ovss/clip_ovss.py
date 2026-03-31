import torch
import torch.nn as nn

from models.builder import build_model
from omegaconf import OmegaConf

from core.rns import RNS_Inference

class CLIP(nn.Module):

    def __init__(self, clip_backbone, class_names, vit_patch_size=16, backbones=["CLIP", "SAM"]):
        super(CLIP, self).__init__()

        self.class_names = class_names
        self.vit_patch_size = vit_patch_size

        # ==== build MaskCLIP backbone =====
        maskclip_cfg = OmegaConf.load(f"configs/{clip_backbone}.yaml")
        self.clip_backbone = build_model(maskclip_cfg["model"], class_names=class_names)

        self.text_query_embeddings = self.clip_backbone.decode_head._get_class_embeddings(self.clip_backbone.backbone, 
                                                                                          class_names).cuda()

        ## ==== build SAM backbone =====
        if "SAM" in backbones:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "./sam2.1_hiera_l.yaml"

            sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False)

            self.mask_generator = SAM2AutomaticMaskGenerator(
                    model=sam2,
                    points_per_side=32,
                    points_per_batch=512,
                    pred_iou_thresh=0.4,
                    stability_score_thresh=0.4,
                    multimask_output=False,
                )
            
        self.test_counter = 0

    def make_input_divisible(self, x: torch.Tensor, pad_value = 0) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=pad_value)
        return x

    @torch.no_grad()
    def get_vlm_features(self, x: torch.Tensor):
        """
        Extracts MaskCLIP features
        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - clip dense features, (torch.Tensor) - output probabilities
        """
        x = self.make_input_divisible(x)
        _, feat, cls = self.clip_backbone(x, return_feat=True)

        return feat, cls
    

class CLIP_OVSS(CLIP):
    
    def __init__(self, clip_backbone, class_names, dataset_key=None, vit_patch_size=16, backbones=["CLIP", "SAM"]):
        super(CLIP_OVSS, self).__init__(clip_backbone, class_names, vit_patch_size, backbones=backbones)

    def forward_pass(self, x: torch.Tensor):
        
        x = self.make_input_divisible(x)
        vlm_proj_feats, _ = self.get_vlm_features(x)

        return vlm_proj_feats

    def forward(self, vlm_features: torch.Tensor):

        return RNS_Inference(self, vlm_features)