import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.TTA_models.Linear import linear_layer

import faiss
import faiss.contrib.torch_utils

import os
import numpy as np

def propose_masks(model, img: torch.Tensor, vlm_features: torch.Tensor):
    
    B, C, hf, wf = vlm_features.shape
            
    vlm_features = vlm_features.reshape(B, C, -1)  # (B, C, H*W)
    # norm the maskclip_feats to unit norm
    vlm_features = vlm_features / vlm_features.norm(dim=1, keepdim=True)

    model.test_counter += 1
        
    if model.mask_proposal == 'SAM':
        
        mask_key = model.dataset_key
        if mask_key == 'context59':
            mask_key = 'context'
        
        if "DINO" in model.__class__.__name__:
            model_name = "dino"
        if "CLIP" in model.__class__.__name__:
            model_name = "clip"

        mask_file = f"./SAM_Masks/{model_name}/{mask_key}/{model.test_counter:05d}.npy"
        if os.path.exists(mask_file):
            masks = np.load(mask_file, allow_pickle=True)
        else:
            masks = model.mask_generator.generate(img.squeeze().permute(1, 2, 0).cpu().numpy())
            os.makedirs(os.path.dirname(mask_file), exist_ok=True)
            np.save(mask_file, masks, allow_pickle=True)

        instance_mask = np.zeros((img.shape[-2], img.shape[-1]), dtype=int)
        if len(masks) != 0:
            sorted_anns = sorted(masks, key=(lambda x: x['area']))  # predicted_iou
            instance_id = 1
            for ann in sorted_anns:
                m = ann['segmentation']
                instance_mask[m] = instance_id
                instance_id += 1
        
        # --- Relabel to remove gaps ---
        unique_ids = np.unique(instance_mask)
        remap = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

        # Vectorized remap
        lut = np.zeros(instance_mask.max() + 1, dtype=int)
        for old_id, new_id in remap.items():
            lut[old_id] = new_id
        instance_mask = lut[instance_mask]

        proposed_masks = torch.from_numpy(instance_mask).to(model.device)

    downsampled_masks = model.make_input_divisible(proposed_masks.unsqueeze(0).unsqueeze(0), pad_value = -1).squeeze()
    downsampled_masks = downsampled_masks.unfold(0, model.vit_patch_size, model.vit_patch_size).unfold(1, model.vit_patch_size, model.vit_patch_size)
    downsampled_masks = downsampled_masks.reshape(hf*wf, -1) # (H*W, vit_patch_size * vit_patch_size)

    mask_indices = torch.arange(0, proposed_masks.max() + 1, dtype=torch.uint8)
    # soft patch pooling weights according to the membership in each superpixel region
    downsampled_masks = torch.cat([((downsampled_masks.unsqueeze(0) == i).sum(dim=-1, keepdim=False) / (downsampled_masks.unsqueeze(0) == i).sum().item())
                                    for i in mask_indices], dim=0).float() # (num_masks, H*W)
    
    # get an average of the features using the masks
    mask_emb = torch.matmul(vlm_features.squeeze(), downsampled_masks.T) # (C, num_masks)
    
    mask_emb = mask_emb.reshape(1, C, -1)
    mask_emb = mask_emb / mask_emb.norm(dim=1, keepdim=True) # (1, C, num_masks)
    
    return mask_emb, proposed_masks

def retrieve_similar_feats(model, query_feats, support_feats, k):    
    support_bank = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), support_feats.shape[1])
    support_bank.add(support_feats.contiguous())
    
    sims, indices = support_bank.search(query_feats.contiguous(), k)

    del support_bank
    
    return sims ,indices

def RNS_Inference(vlm, vlm_features: torch.Tensor):

        if vlm.mask_proposal == 'None':
            B, C, hf, wf = vlm_features.shape
        else:
            B, C, nm = vlm_features.shape
                
        vlm_features = vlm_features.reshape(B, C, -1)  # (B, C, H*W)
        # norm the maskclip_feats to unit norm
        vlm_features = vlm_features / vlm_features.norm(dim=1, keepdim=True)
        
        if vlm.use_text:
            if vlm.drop_text_fraction > 0.0:
                global_textual_class_scores = F.softmax(torch.ones(vlm.num_classes, device=vlm_features.device), dim=0)
            else:
                glob_emb = (vlm_features.permute(0, 2, 1).squeeze(0).mean(dim=0))
                glob_emb = glob_emb / glob_emb.norm()
                global_textual_class_scores = F.softmax((glob_emb
                                                        @ vlm.textual_embeddings.T) * vlm.class_score_Temp, dim=0)
        else:
            global_textual_class_scores = F.softmax(torch.ones(vlm.num_classes, device=vlm_features.device), dim=0)
            
        if global_textual_class_scores.shape[0] > vlm.num_classes:
            global_textual_class_scores = vlm.reduce_to_true_classes(global_textual_class_scores.unsqueeze(0)).squeeze(0)

        class_embeddings = vlm.textual_embeddings
        class_preds = vlm_features.permute(0, 2, 1).squeeze(0) @ class_embeddings.T  # (H*W, num_classes)
        
        # softmax with temp
        soft_text_preds = F.softmax(class_preds * 100, dim=1) # (H*W, num_classes)
        test_labels = soft_text_preds.argmax(dim=1)  # (H*W, )

        support_vlm_feats = vlm.support_vlm_feats

        #### RETRIEVAL ####
        supp_sims, supp_indices = retrieve_similar_feats(vlm, (+1) * vlm_features.permute(0, 2, 1).squeeze(0), 
                                                         support_vlm_feats, vlm.k) # (H*W, k)

        unique_supp_indices, inverse_supp_indices = torch.unique(supp_indices, sorted=True, return_inverse=True)

        # remove -1 from unique_supp_indices
        unique_supp_indices = unique_supp_indices[unique_supp_indices != -1]

        support_feats = vlm.support_vlm_feats[unique_supp_indices].cuda() # (M, C)        
        
        # get the labels of the supp features
        support_labels = unique_supp_indices % vlm.num_classes # (M, )
        
        # No filter
        filter = torch.ones(support_feats.shape[0], device=support_feats.device, dtype=torch.bool)  # (M, )
        
        # filter out drop_classes
        filter = ~torch.isin(support_labels, vlm.drop_classes)  # (M, )
        
        support_feats = support_feats[filter]  # (M', C)
        support_labels = support_labels[filter]  # (M', )
        
        num_retrieved_instances = support_feats.shape[0]
        
        # Average patch embeddings on text classes
        potential_text_labels = test_labels.unique()
        
        pseudo_instances = []
        pseudo_labels = []
        for i in potential_text_labels:
            
            feats = vlm_features.permute(0, 2, 1).squeeze(0)[test_labels == i]
            
            pseudo_instances.append(feats.mean(dim=0, keepdim=True))
            pseudo_labels.append(torch.tensor([i], device=feats.device))
        
        pseudo_instances = torch.cat(pseudo_instances, dim=0)
        pseudo_instances = pseudo_instances / pseudo_instances.norm(dim=1, keepdim=True)  # normalize     
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        
        filter = torch.isin(pseudo_labels, vlm.drop_classes)
        pseudo_instances = pseudo_instances[filter]
        pseudo_labels = pseudo_labels[filter]

        text_pseudo_aug_feats = []
        text_pseudo_aug_labels = []
        for lambda_ in [0.9, 0.8, 0.6, 0.4, 0.2, 0.0]:

            feats = lambda_ * pseudo_instances + (1 - lambda_) * vlm.textual_embeddings[pseudo_labels]
            feats = feats / feats.norm(dim=1, keepdim=True)  # normalize
            
            text_pseudo_aug_feats.append(feats)
            text_pseudo_aug_labels.append(pseudo_labels)
            
        text_pseudo_aug_feats = torch.cat(text_pseudo_aug_feats, dim=0)
        text_pseudo_aug_labels = torch.cat(text_pseudo_aug_labels, dim=0)  
        text_pseudo_aug_labels = vlm.mixed_labels[text_pseudo_aug_labels]
        
        num_pseudo_instances = text_pseudo_aug_feats.shape[0]
            
        mixed_class_aug_feats = vlm.mixed
        mixed_class_aug_labels = vlm.mixed_labels.repeat(vlm.mixed.shape[0] // vlm.mixed_labels.shape[0])
        
        filter = torch.isin(mixed_class_aug_labels, support_labels.unique())
        filter = filter & torch.isfinite(mixed_class_aug_feats).all(dim=1)
        mixed_class_aug_feats = mixed_class_aug_feats[filter]
        mixed_class_aug_labels = mixed_class_aug_labels[filter]
        
        num_class_embeddings = mixed_class_aug_feats.shape[0]
        
        # one hot encode support_labels and mixed_class_aug_labels
        support_labels = F.one_hot(support_labels, num_classes=vlm.num_classes).float()
        mixed_class_aug_labels = F.one_hot(mixed_class_aug_labels, num_classes=vlm.num_classes).float()
        
        # soft pseudo label based on the cosine similarity with the textual embeddings
        text_pseudo_aug_labels = torch.matmul(text_pseudo_aug_feats, vlm.textual_embeddings.T) 
        text_pseudo_aug_labels = F.softmax(text_pseudo_aug_labels * 100, dim=1)
        
        if vlm.textual_embeddings.shape[0] > vlm.num_classes:
            
            text_pseudo_aug_labels = vlm.reduce_to_true_classes(text_pseudo_aug_labels)

        support_feats = torch.cat([support_feats, mixed_class_aug_feats, text_pseudo_aug_feats], dim=0)
        support_labels = torch.cat([support_labels, mixed_class_aug_labels, text_pseudo_aug_labels], dim=0)
        support_labels_weights = global_textual_class_scores[support_labels.argmax(dim=1)]
        
        # L1 norm of the weights
        support_labels_weights = support_labels_weights / support_labels_weights.sum()
        
        M = support_feats.shape[0]
        HW = vlm_features.shape[2]
        
        X = torch.cat([vlm_features.permute(0, 2, 1).squeeze(0), support_feats], dim=0) # (HW + M, C)

        model = linear_layer(in_features=vlm_features.shape[1], out_features=vlm.num_classes, dropout=0.0,
                            init_weights=None, bias=True).to(vlm.device)
        
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=vlm.lr, weight_decay=5e-4)
        cross_entropy = nn.CrossEntropyLoss(reduction='none')
        kl_div = nn.KLDivLoss(reduction='none')
    
        # Training loop
        with torch.enable_grad():
            for epoch in range(vlm.epochs):

                optimizer.zero_grad()
            
                # Linear
                output = model(X[HW:])  # Forward pass on the whole support embs
                
                output_log = F.log_softmax(output, dim=-1)
                loss_pseudo = kl_div(output_log[(num_retrieved_instances + num_class_embeddings):],
                                    support_labels[(num_retrieved_instances + num_class_embeddings):])
                loss_pseudo = (loss_pseudo * support_labels_weights[(num_retrieved_instances + num_class_embeddings):].unsqueeze(1)).sum(dim=-1)
                
                loss_CE = cross_entropy(output[:(num_retrieved_instances + num_class_embeddings)],
                                        support_labels[:(num_retrieved_instances + num_class_embeddings)].argmax(dim=-1).to(torch.long))
                loss_CE = (loss_CE * support_labels_weights[:(num_retrieved_instances + num_class_embeddings)])

                loss_mixed = loss_CE[num_retrieved_instances:]
                loss_retrieved = loss_CE[:num_retrieved_instances]
                    
                loss = torch.cat([loss_retrieved, vlm.beta_mixed * loss_mixed, vlm.beta_pseudo * loss_pseudo]).mean()
                
                loss.backward()
                optimizer.step()
        
        # Inference for test patches
        model.eval()
        with torch.no_grad():
            # Linear
            output_rns = model(X[:HW])

        return output_rns