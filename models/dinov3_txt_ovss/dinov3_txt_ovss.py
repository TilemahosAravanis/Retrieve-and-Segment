import torch
import torch.nn as nn

from core.rns import RNS_Inference

class DINOV3_TXT(nn.Module):

    def __init__(self, clip_model, class_names, vit_patch_size=16, backbones=["DINO", "SAM"]):    
        super(DINOV3_TXT, self).__init__()
        
        self.class_names = class_names
        self.vit_patch_size = vit_patch_size

        # PLEASE FILL IN YOUR PERSONAL DOWNLOAD URLs (check https://github.com/facebookresearch/dinov3)
        backbone = ""
        weights = ""

        import torch
        # DINOv3
        self.model, self.tokenizer = torch.hub.load('./dinov3', 'dinov3_vitl16_dinotxt_tet1280d20h24l', source='local', 
                                                    weights=weights, backbone_weights=backbone)
        self.model = self.model.cuda()
        
        from .prompt_templates import imagenet_templates
        self.class_name_templates = imagenet_templates

        temp, class_indices = [], []
        for idx in range(len(class_names)):
            names_i = class_names[idx].split(';')
            temp += list(names_i)
            class_indices += [idx for _ in range(len(names_i))]
            class_names = [item.replace('\n', '') for item in class_names]
            
        class_names = temp

        query_features = []
        cls_aligned_feats = []
        with torch.no_grad():
            for name in class_names:

                query = self.tokenizer.tokenize([temp.format(name) for temp in imagenet_templates])
                query = query.to("cuda")
                feature = self.model.encode_text(query)
                cls_aligned_feat = feature

                feature = feature[:, feature.shape[1] // 2 :]
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()

                cls_aligned_feat /= cls_aligned_feat.norm()
                cls_aligned_feat = cls_aligned_feat.mean(dim=0)
                cls_aligned_feat /= cls_aligned_feat.norm()

                query_features.append(feature.unsqueeze(0))
                cls_aligned_feats.append(cls_aligned_feat.unsqueeze(0))

        query_features = torch.cat(query_features, dim=0)
        cls_aligned_feats = torch.cat(cls_aligned_feats, dim=0)

        self.text_query_embeddings = query_features
        self.cls_aligned_embeddings = cls_aligned_feats
        
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
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        from torchvision.transforms import v2
        x = v2.Normalize(mean=mean, std=std)(x)

        img_shape = x.shape
        B, _, H, W = x.shape

        with torch.autocast('cuda', dtype=torch.float):
            with torch.no_grad():
                image_class_token, image_patch_tokens, backbone_patch_tokens = self.model.encode_image_with_patch_tokens(x)
        
        dinov3txt_feats = backbone_patch_tokens.permute(0, 2, 1).reshape(B, -1, H // self.vit_patch_size, W // self.vit_patch_size)
        
        image_class_token = image_class_token.squeeze()

        return dinov3txt_feats, image_class_token
    

class DINOV3_TXT_OVSS(DINOV3_TXT):

    def __init__(self, clip_model, class_names, dataset_key=None, vit_patch_size=16, backbones=["DINO", "SAM"]):

        super(DINOV3_TXT_OVSS, self).__init__(clip_model, class_names, vit_patch_size, backbones)

    def forward_pass(self, x: torch.Tensor):
        
        x = self.make_input_divisible(x)
        vlm_proj_feats, _ = self.get_vlm_features(x)

        return vlm_proj_feats
    
    def forward(self, vlm_features: torch.Tensor):

        return RNS_Inference(self, vlm_features)