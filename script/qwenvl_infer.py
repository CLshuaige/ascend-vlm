import os
import sys
import time
import torch
import numpy as np
import requests
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from transformers import Qwen2VLConfig, AutoTokenizer

#shutil.copy("/home/chenl/project/ascend/ascend-llm/export_llama/modeling_qwen2_vl_m.py", "/home/chenl/miniconda3/envs/ascend-llm/lib/python3.9/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py")
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
## imports for PACT
from utils import *
import math

print(ort.get_device())
device='cuda:1'
# Command line arguments
model_path = sys.argv[1]
#onnx_path = sys.argv[2]
PACT = True
datatype = np.float16
sim = True
# paths = {
#     "llm.onnx": "/home/chenl/project/ascend/llm-export/script/model_1024/fixed_llm/llm.onnx",
#     "visual.onnx": "/home/chenl/project/ascend/llm-export/script/model_1024/onnx/visual.onnx",
#     "embedder.onnx": "/home/chenl/project/ascend/llm-export/script/model_1024/onnx/embedder.onnx",
# }

# Initialize model config and tokenizer
model_config = Qwen2VLConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Model configuration
kvcache_len = 1024
num_attention_heads = model_config.num_attention_heads
num_key_value_heads = model_config.num_key_value_heads
head_dim = model_config.hidden_size // num_attention_heads
num_layers = model_config.num_hidden_layers

# Setup ONNX sessions
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

onnx_names = ["llm.onnx", "embedder.onnx", "visual.onnx"]
if PACT:
    onnx_names = ["llm1.onnx", "llm2.onnx", "embedder.onnx", "visual.onnx"]
onnx_names = onnx_names[:]
print(onnx_names)
#model_paths = {m: os.path.join(onnx_path, m) for m in onnx_names}
model_paths = {
    "llm.onnx": "/home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/int8_1024/llm/llm.onnx",
    "embedder.onnx": "/home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/fp16_1024_split/embedder/embedder.onnx",
    "visual.onnx": "/home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/fp16_1024_split/visual/visual.onnx"
}
if PACT:
    model_paths = {
        "llm1.onnx": "/home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/int8_pact/llm_1_4.onnx",
        "llm2.onnx": "/home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/int8_pact/llm_5_28.onnx",
        "embedder.onnx": "/home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/fp16_1024_split/embedder/embedder.onnx",
        "visual.onnx": "/home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/int8_pact/visual/visual_dy.onnx"
    }

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
def pact(hidden_states, query_states_before_rope, key_states_before_rope, key_states, query_states, image_mask, real_position_ids):
    from types import SimpleNamespace
    pact_config = {
        "visual_token_reduction": True, ## PACT
        "layer_for_reduction": 4,
        "progessive_reduction": False,
        "use_DBDPC": True, ## PACT
        #"cutoff": 0.21, ## PACT
        "cutoff": 0.06,
        "vector_to_use_in_distance_clustering": "current_k_cosine",
        "take_mean": True,
        "include_pruned_in_mean": True,
        "do_not_consider_non_image_tokens_as_pruned": True,
        "coef_pruned": 1.5,
        "avoid_numerical_instability_DBDPC": True,
        "withdraw_visual_tokens": False,
        "VTW_equivalant_layer_for_reduction": -1,
        "equivalent_reduc_percentage_vtw": 0.0,
        "use_tome": False,
        "perc_tokeep_tome_total": 1.0,
        "tome_equivalant_layer_for_reduction": 4,
        "use_kmeans": False,
        "perc_tokeep_kmeans": 1.0,
        "use_dpc": False,
        "percentage_to_keep_dpc": 1.0,
        "use_agglomerative": False,
        "percentage_to_keep_agglomerative": 1.0,
        "linkage": "single",
        "use_dbscan": False,
        "eps_dbscan": 0.1,
        "noise_as_clusters_dbscan": False,
        "token_pruning": True,  ## PACT
        "use_all_non_text_pruning": True,
        "prune_with_norm": False,
        "use_cosine_in_token_pruning": False,
        "use_attention_in_token_pruning": False,
        "use_mask_in_use_attention_in_token_pruning": False,
        "use_IQR_in_token_pruning": False,
        "alpha_IQR": 0.5,
        "pruning_filter_wth_percentage": True,
        #"pruning_tokeep_percentage_value": 0.55,   ##PACT
        "pruning_tokeep_percentage_value": 0.55,
        "multiply_by_norm": True,  ## PACT
        "norm_to_use": 2,
        "avoid_numerical_instability_prune": True,
        "no_proportional_attention": False,
        "change_position_ids": False,
        "get_mean_position_id": False,
        "synchro": False,
        "need_kq": True,
        "do_not_upcast_to_full_precision_for_pruning": False,
        "keep_casual": True,
        "get_performance_metrics": False,
        "get_reduction_ratio": True,
        "use_custom_merging": False,    ##PACT
        "use_custom_pruning": False,
        "log_output_path": "agg_pact_logs", ## PACT
    }
    pact_config = SimpleNamespace(**pact_config)

    # to tensor
    # input: B, L, N, D
    if not isinstance(image_mask, torch.Tensor):
        image_mask = torch.tensor(image_mask, dtype=torch.bool)
    if not isinstance(hidden_states, torch.Tensor):
        hidden_states = torch.tensor(hidden_states, dtype=torch.float)
    if not isinstance(query_states_before_rope, torch.Tensor):
        query_states_before_rope = torch.tensor(query_states_before_rope, dtype=torch.float).transpose(1, 2)
    if not isinstance(key_states_before_rope, torch.Tensor):
        key_states_before_rope = torch.tensor(key_states_before_rope, dtype=torch.float).transpose(1, 2)
    if not isinstance(key_states, torch.Tensor):
        key_states = torch.tensor(key_states, dtype=torch.float).permute(0,2,1,3).reshape(1,-1,256)

    if not isinstance(real_position_ids, torch.Tensor):
        real_position_ids = torch.tensor(real_position_ids, dtype=torch.int64)
    # Shape: B, N, L, D
    if image_mask.dim() == 2:
        image_mask = image_mask.squeeze(0)

    image_in_input = True
    if pact_config.visual_token_reduction :
        is_reduction_layer_or_after=True
        is_reduction_layer=True

        if image_in_input and is_reduction_layer: 
            
            if pact_config.get_performance_metrics :
                torch.cuda.synchronize()
                start_algo=time.time()
            is_image=image_mask
            ###
            if  pact_config.need_kq :
                last_true_idx = torch.where(is_image)[0].max()
                is_image_indices = torch.where(is_image)[0]
                if is_image_indices.numel() > 0:
                    last_true_idx = is_image_indices.max()
                else:
                    last_true_idx = -1

                is_not_text = torch.zeros_like(is_image, dtype=torch.float)
                is_not_text[:last_true_idx + 1] = 1
                is_not_text=is_not_text.bool()
                
            
            reduction=[0,0]

            if  pact_config.need_kq :
                current_k,current_q,current_k_cosine,current_q_cosine=key_states_before_rope,query_states_before_rope,key_states,query_states
                vector_to_use_in_distance_clustering=eval(pact_config.vector_to_use_in_distance_clustering)


            ## pruning based reduction starts here
            if pact_config.token_pruning :
                if pact_config.use_cosine_in_token_pruning :
                    if pact_config.use_all_non_text_pruning :
                        current_k_image=current_k_cosine[:,is_not_text]
                    else :
                        current_k_image=current_k_cosine[:,is_image]
                    current_q_image=current_q_cosine[:,is_not_text]
                else :
                    if pact_config.use_all_non_text_pruning :
                        current_k_image=current_k[:,is_not_text]
                    else :
                        current_k_image=current_k[:,is_image]
                    current_q_image=current_q[:,is_not_text]
                bsz, q_len, _, _ = current_q_image.size()

                num_key_value_heads = 2
                head_dim = 128
                num_heads = 12
                num_key_value_groups = num_heads // num_key_value_heads
                current_k_image = current_k_image.view(bsz, -1, num_key_value_heads, head_dim).transpose(1, 2)
                current_k_image = repeat_kv(current_k_image, num_key_value_groups)
                current_q_image = current_q_image.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

                if pact_config.use_custom_pruning  :
                    pass
                    # scores=custom_pruning(current_k_image,current_q_image)
                    # if pact_config.use_all_non_text_pruning :
                    #     scores=scores[:,is_image[is_not_text]]
                else :
                    if pact_config.use_attention_in_token_pruning :
                        #for fastv
                        global_q=current_q_image
                    else :
                        #for pact
                        global_q=torch.mean(current_q_image,dim=2,keepdim=True)
                    if pact_config.do_not_upcast_to_full_precision_for_pruning :
                        scores=torch.matmul(global_q/math.sqrt(head_dim,),current_k_image.transpose(-2,-1))
                    else :
                        scores=torch.matmul(global_q.to(torch.float32)/math.sqrt(head_dim,),current_k_image.to(torch.float32).transpose(-2,-1))
                    if pact_config.use_attention_in_token_pruning and pact_config.use_mask_in_use_attention_in_token_pruning:
                        causal_mask_att_calculation = torch.triu(torch.ones((q_len, q_len), device=global_q.device), diagonal=1).bool()
                        if not pact_config.use_all_non_text_pruning :
                            causal_mask_att_calculation=causal_mask_att_calculation[:,is_image[is_not_text]]
                        scores = scores.masked_fill(causal_mask_att_calculation, float('-inf'))

                    scores=torch.softmax(scores,dim=-1,dtype=torch.float32).mean(1)
                    scores = torch.nan_to_num(scores, nan=0.0)
                    
                    if pact_config.use_attention_in_token_pruning :
                        scores=scores.mean(-2)
                    else :    
                        scores=scores.squeeze(-2)
                    if pact_config.use_all_non_text_pruning :
                        scores=scores[:,is_image[is_not_text]]
                
                    if pact_config.multiply_by_norm :
                        norm=torch.norm(hidden_states[:,is_image].to(torch.float32), dim=-1,p=pact_config.norm_to_use).squeeze()
                        scores=scores.to(torch.float32)*norm

                if pact_config.avoid_numerical_instability_prune and pact_config.pruning_filter_wth_percentage:
                    scores= scores.squeeze()
                    sorted_indices = torch.argsort(scores)
                    ranks = torch.empty_like(sorted_indices).to(scores.device).to(scores.dtype)
                    ranks[sorted_indices] = torch.arange(len(scores)).to(scores.device).to(scores.dtype)
                    scores=ranks

                if pact_config.pruning_filter_wth_percentage :
                    scores= scores.squeeze()
                    num_elements = scores.numel()
                    num_to_keep = int(num_elements * pact_config.pruning_tokeep_percentage_value)
                    sorted_scores, _ = torch.sort(scores, descending=True)
                    thresh_scores = sorted_scores[num_to_keep - 1]
                elif pact_config.use_IQR_in_token_pruning :
                    pass
                    # q1 = torch.quantile(scores_flat, 0.25)
                    # q3 = torch.quantile(scores_flat, 0.75)
                    # thresh_scores = q1 + pact_config.alpha_IQR * (q3 - q1)

                first_mask = (scores >= thresh_scores).squeeze().bool()
                                                
            elif pact_config.prune_with_norm :
                norm=torch.norm(hidden_states[:,is_image].to(torch.float32), dim=-1,p=2).squeeze()
                scores= norm.squeeze()
                num_elements = scores.numel()
                num_to_keep = int(num_elements * pact_config.pruning_tokeep_percentage_value)
                sorted_scores, _ = torch.sort(scores, descending=True)
                thresh_scores = sorted_scores[num_to_keep - 1]
                first_mask = (scores > thresh_scores).squeeze().bool()

            elif pact_config.withdraw_visual_tokens :
                first_mask = torch.zeros_like(is_image.nonzero(), dtype=torch.bool).squeeze()

            else :
                first_mask=torch.ones_like(is_image.nonzero()).squeeze().bool()

            if pact_config.synchro or pact_config.get_reduction_ratio :
                reduction[0]+=first_mask.shape[0]
            first_mask_global=is_image.clone()
            first_mask_global[is_image]=first_mask.clone()


            ## clustering based reduction starts here
            
            if pact_config.use_DBDPC:
                #real_position_ids 3,1,seqlen
                real_position_ids_after_mask_image=real_position_ids.permute(2,1,0)[first_mask_global]  #N,1,3
                if not pact_config.include_pruned_in_mean :
                    merged,second_mask,weights,position_ids_after_reduction=token_reduction(hidden_states[:,first_mask_global].squeeze(0),vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0),position_ids=real_position_ids_after_mask_image,reduction=None,cutoff=pact_config.cutoff,pact_config=pact_config)
                else :
                    if pact_config.do_not_consider_non_image_tokens_as_pruned :
                        merged,second_mask,weights,position_ids_after_reduction=token_reduction(hidden_states[:,first_mask_global].squeeze(0),vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0),position_ids=real_position_ids_after_mask_image,reduction=None,cutoff=pact_config.cutoff,pruned_hiddens=hidden_states[:,is_image][:,~first_mask].squeeze(0),pruned_keys=vector_to_use_in_distance_clustering[:,is_image][:,~first_mask].squeeze(0),pact_config=pact_config)
                    else :
                        merged,second_mask,weights,position_ids_after_reduction=token_reduction(hidden_states[:,first_mask_global].squeeze(0),vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0),position_ids=real_position_ids_after_mask_image,reduction=None,cutoff=pact_config.cutoff,pruned_hiddens=hidden_states[:,~first_mask_global].squeeze(0),pruned_keys=vector_to_use_in_distance_clustering[:,~first_mask_global].squeeze(0),pact_config=pact_config)
                if pact_config.get_mean_position_id:
                    position_ids_after_reduction=position_ids_after_reduction.to(real_position_ids.dtype).permute(2,1,0)
                    position_ids[:3,:,:][:,:,first_mask_global]=position_ids_after_reduction
                    real_position_ids[:,:,first_mask_global]=position_ids_after_reduction

                hidden_states[:,first_mask_global]=merged.unsqueeze(0)
                second_mask=second_mask.squeeze()
                if pact_config.synchro or pact_config.get_reduction_ratio:
                    reduction[1]+=second_mask.sum().item()

            elif pact_config.use_custom_merging :
                real_position_ids_after_mask_image=real_position_ids.permute(2,1,0)[first_mask_global]  #N,3,1
                
                if not pact_config.include_pruned_in_mean :
                    merged,second_mask,weights,position_ids_after_reduction=custom_token_reduction(hidden_states[:,first_mask_global].squeeze(0), vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0), position_ids=real_position_ids_after_mask_image, cutoff=pact_config.cutoff)
                else :
                    merged,second_mask,weights,position_ids_after_reduction=custom_token_reduction(hidden_states[:,first_mask_global].squeeze(0), vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0), position_ids=real_position_ids_after_mask_image, cutoff=pact_config.cutoff,pruned_hiddens=hidden_states[:,is_image][:,~first_mask].squeeze(0),pruned_for_reduction=vector_to_use_in_distance_clustering[:,is_image][:,~first_mask].squeeze(0))
                
                position_ids_after_reduction=position_ids_after_reduction.to(real_position_ids.dtype).permute(2,1,0)
                position_ids[:3,:,:][:,:,first_mask_global]=position_ids_after_reduction
                real_position_ids[:,:,first_mask_global]=position_ids_after_reduction
                hidden_states[:,first_mask_global]=merged.unsqueeze(0)
                second_mask=second_mask.squeeze()
                if pact_config.synchro or pact_config.get_reduction_ratio:
                    reduction[1]+=second_mask.sum().item()
            elif pact_config.use_tome :
                pass
                # if index>=1 :
                #     sizes=torch.exp(self.weights[index-1].squeeze()[first_mask_global]) #we store weights as log values so need to revert them back
                # else :
                #     sizes=torch.ones_like(hidden_states[:,first_mask_global].squeeze()[:,0])
                # n_layers=len(self.layers)
                # if index==0 :
                #     effective_percentage_to_keep=1-(1-pact_config.perc_tokeep_tome_total)*((n_layers-pact_config.tome_equivalant_layer_for_reduction)/n_layers)
                #     r_intial=6*n_layers*(1-effective_percentage_to_keep)/((n_layers+1)*(2*n_layers+1)) *(is_image.sum())
                # r=int(r_intial*((n_layers-index)/n_layers))+1
                # merged,second_mask,weights,_=token_reduction_tome(hidden_states[:,first_mask_global].squeeze(0),vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0), sizes,reduction=self.reduction,r=r)
                # if pact_config.take_mean :
                #     hidden_states[:,first_mask_global]=merged.unsqueeze(0)
                # second_mask=second_mask.squeeze()              
            elif pact_config.use_kmeans :
                pass
                # k=int(pact_config.perc_tokeep_kmeans*is_image.sum())
                # merged,second_mask,weights,_=token_reduction_kmeans(hidden_states[:,first_mask_global].squeeze(0),vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0), k=k)
                # hidden_states[:,first_mask_global]=merged.unsqueeze(0)
                # second_mask=second_mask.squeeze()
                # if pact_config.synchro or pact_config.get_reduction_ratio:
                #     self.reduction[1]+=second_mask.sum().item()
            elif pact_config.use_dpc :
                pass
                # merged,second_mask,weights,_=token_reduction_dpc(hidden_states[:,first_mask_global].squeeze(0),vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0), pact_config.percentage_to_keep_dpc,reduction=self.reduction)
                # hidden_states[:,first_mask_global]=merged.unsqueeze(0)
                # second_mask=second_mask.squeeze()
                # if pact_config.synchro or pact_config.get_reduction_ratio:
                #     self.reduction[1]+=second_mask.sum().item()
            elif pact_config.use_dbscan :
                pass
                # merged,second_mask,weights,_=token_reduction_dbscan(hidden_states[:,first_mask_global].squeeze(0),vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0), eps=pact_config.eps_dbscan,isolate_noise_as_clusters=pact_config.noise_as_clusters_dbscan,reduction=self.reduction)
                # hidden_states[:,first_mask_global]=merged.unsqueeze(0)
                # second_mask=second_mask.squeeze()
                # if pact_config.synchro or pact_config.get_reduction_ratio:
                #     self.reduction[1]+=second_mask.sum().item()
            elif pact_config.use_agglomerative :
                pass
                # merged,second_mask,weights,_=token_reduction_agglomerative(hidden_states[:,first_mask_global].squeeze(0),vector_to_use_in_distance_clustering[:,first_mask_global].squeeze(0), pact_config.percentage_to_keep_agglomerative,reduction=self.reduction,linkage=pact_config.linkage)
                # hidden_states[:,first_mask_global]=merged.unsqueeze(0)
                # second_mask=second_mask.squeeze()
                # if pact_config.synchro or pact_config.get_reduction_ratio:
                #     self.reduction[1]+=second_mask.sum().item()
            else :
                pass
                # second_mask=torch.ones_like(first_mask_global.nonzero()).squeeze().bool()
                # if index==pact_config.layer_for_reduction :
                #     weights=torch.ones_like(second_mask).to(hidden_states.dtype)
                # else :
                #     weights=self.weights[index-1].squeeze()[first_mask_global]
                # if pact_config.synchro or pact_config.get_reduction_ratio:
                #     self.reduction[1]+=second_mask.sum().item()

            weights_final=torch.ones_like(is_image).to(torch.float16)
            weights_final[first_mask_global]=weights.to(torch.float16)
            weights=weights_final.unsqueeze(0)

            mask_final=torch.ones_like(is_image).bool()
            mask_final[is_image]= first_mask
            mask_final[first_mask_global]=second_mask
            
            #position_ids=position_ids[:,:,mask_final]
            hidden_states=hidden_states[:,mask_final]
            real_position_ids=real_position_ids[:,:,mask_final]
            #cache_position=cache_position[mask_final.squeeze()]
            #position_embeddings=(position_embeddings[0][:,:,mask_final],position_embeddings[1][:,:,mask_final])
            weights = weights[:,mask_final]
            weights = torch.log(weights)

            if pact_config.need_kq :
                is_not_text=is_not_text[mask_final]
            is_image=is_image[mask_final]
            if pact_config.progessive_reduction :
                pass
                # if index==pact_config.layer_for_reduction :
                #     self.weights=dict()
                # self.weights[index]=weights
            else :
                weights=weights

            weights_forward=weights

            seq_len = real_position_ids.shape[2]
            weights_forward=weights_forward.squeeze().unsqueeze(0).repeat(seq_len,1)
            
            weights_forward=weights_forward.to(hidden_states.dtype).unsqueeze(0).unsqueeze(0)
            if pact_config.get_performance_metrics :
                torch.cuda.synchronize() 
                total_algo_time+=time.time()-start_algo

        elif is_reduction_layer : 
            # if pact_config.progessive_reduction :
            #     self.weights[index]=torch.cat((self.weights[index],torch.zeros(self.weights[index].size(0), hidden_states.shape[1], device=self.weights[index].device)),dim=-1)
            #     weights_forward=self.weights[index]
            # else :
            #     self.weights=torch.cat((self.weights,torch.zeros(self.weights.size(0), hidden_states.shape[1], device=self.weights.device)),dim=-1)
            #     weights_forward=self.weights
            
            seq_len = real_position_ids.shape[2]
            weights_forward=weights_forward.squeeze().unsqueeze(0).repeat(seq_len,1)
            weights_forward=weights_forward.to(hidden_states.dtype).unsqueeze(0).unsqueeze(0)

    if pact_config.visual_token_reduction and pact_config.synchro and image_in_input:
        pass
        #self.mean_visual_tokens_all[0]+=position_ids[3:,:,:].bool().squeeze().sum().item()

    if weights_forward is not None and pact_config.no_proportional_attention :
        weights_forward=None

    if pact_config.change_position_ids and image_in_input :
        batch_size, seq_len = real_position_ids.shape
        real_position_ids[:] = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).to(real_position_ids.device).to(real_position_ids.dtype)

    position_ids = real_position_ids
    hidden_states = hidden_states

    return hidden_states, position_ids, weights, reduction


sessions = {m: ort.InferenceSession(path, 
                                sess_options=session_options, 
                                providers=[
                                    ("CUDAExecutionProvider", {"device_id": 1,}), 
                                            "CPUExecutionProvider"]) 
                                for m, path in model_paths.items()}

inputs = {}
outputs = {}
for m in onnx_names:
    inputs[m] = [sessions[m].get_inputs()[i].name for i in range(len(sessions[m].get_inputs()))]
    outputs[m] = [sessions[m].get_outputs()[i].name for i in range(len(sessions[m].get_outputs()))]
print(inputs)
print(outputs)

def preprocess_image(images):

    temporal_patch_size = 2
    patch_size =14
    merge_size = 1
    images = [images] * temporal_patch_size
    #patches = torch.concat(images, axis=0)
    # for numpy.ndarray
    patches = np.concatenate(images, axis=0)
    _, channel, height, width = patches.shape
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    #patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    # for numpy.ndarray
    patches = np.transpose(patches, (0, 3, 6, 4, 7, 2, 1, 5, 8))
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    image_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)

    # rotary_pos_emb = rot_pos_emb(image_grid_thw)
    # emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    # position_embeddings = (emb.cos(), emb.sin())
    return flatten_patches, image_grid_thw
visual = True
len_image_embeds = 0
if visual:
    # Process image
    image_url = '/home/chenl/project/ascend/llm-export/script/demo.jpeg'

    w, h = 420, 420
    if w:
        image = Image.open(image_url).resize((w, h)).convert('RGB')
    #image = Image.open(BytesIO(requests.get(image_url).content)).resize((w, h)).convert('RGB')
    else:
        image = Image.open(image_url).convert('RGB')
    image_array = np.expand_dims(np.transpose(np.array(image).astype(np.float32), (2, 0, 1)), axis=0) / 255.0
    pixel_values = preprocess_image(image_array)[0].astype(datatype)
    pixel_values = pixel_values[np.newaxis, :, :]

    # Run visual model
    visual_embeds = sessions['visual.onnx'].run(
        outputs['visual.onnx'],
        dict(zip(inputs['visual.onnx'],
            [pixel_values]))
    )[0]
    visual_embeds = visual_embeds[np.newaxis, :, :]
    len_image_embeds = visual_embeds.shape[1]

prompt = "Is there a dog in the image?"


formatted_prompt = f"\n<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * len_image_embeds}<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer(formatted_prompt, return_tensors='pt')['input_ids']
image_mask = (input_ids == 151655)
input_lengths = np.array([input_ids.shape[1]], dtype=np.int64)
# tokens = np.zeros(max_length, dtype=np.int64)
# tokens[:input_ids.shape[1]] = input_ids[0, :]

# tokens = input_ids[0, :].numpy()
# text_embeds = sessions['embedder.onnx'].run(
#     outputs['embedder.onnx'],
#     dict(zip(inputs['embedder.onnx'],
#         [tokens]))
# )
tokens = input_ids.to(device)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
#     device_map=device,
# )
# model.eval()
# embeddings = model.model.embed_tokens(tokens).to(torch.float16)
# embeddings = embeddings.cpu().detach().numpy()
def get_embeddings(tokens):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().detach().numpy()
    # make the tokens from [785] to [[785]]
    if isinstance(tokens, np.int64):
        tokens = np.array([[tokens]])
    embeddings = sessions['embedder.onnx'].run(
        outputs['embedder.onnx'],
        dict(zip(inputs['embedder.onnx'],
            [tokens]))
    )
    return embeddings[0]
embeddings = get_embeddings(tokens)
print(embeddings.shape)
# fill the embeddings with visual_embeds following the image_mask
if visual:
    embeddings[image_mask] = visual_embeds[0, :, :]

pre_embeds_len = embeddings.shape[1]

if PACT:
    print(f"attention_mask input shape: {sessions['llm1.onnx'].get_inputs()[0].shape}, type: {sessions['llm1.onnx'].get_inputs()[0].type}")
    print(f"position_ids input shape: {sessions['llm1.onnx'].get_inputs()[1].shape}, type: {sessions['llm1.onnx'].get_inputs()[1].type}")
    print(f"past_key_values input shape: {sessions['llm1.onnx'].get_inputs()[2].shape}, type: {sessions['llm1.onnx'].get_inputs()[2].type}")
    print(f"input_embeds input shape: {sessions['llm1.onnx'].get_inputs()[3].shape}, type: {sessions['llm1.onnx'].get_inputs()[3].type}")

    print(f"attention_mask input shape: {sessions['llm2.onnx'].get_inputs()[0].shape}, type: {sessions['llm2.onnx'].get_inputs()[0].type}")
    print(f"position_ids input shape: {sessions['llm2.onnx'].get_inputs()[1].shape}, type: {sessions['llm2.onnx'].get_inputs()[1].type}")
    print(f"past_key_values input shape: {sessions['llm2.onnx'].get_inputs()[2].shape}, type: {sessions['llm2.onnx'].get_inputs()[2].type}")
    print(f"input_embeds input shape: {sessions['llm2.onnx'].get_inputs()[3].shape}, type: {sessions['llm2.onnx'].get_inputs()[3].type}")
#print(f"attention_mask input shape: {sessions['llm.onnx'].get_inputs()[0].shape}, type: {sessions['llm.onnx'].get_inputs()[0].type}")
def get_attention_mask(input_length, cache_pos=-1, kvcache_len=1024, dim4=True):
    # if gen_len == 0:
    #     attention_mask = np.ones((1, 1, input_lengths, input_lengths), dtype=np.float32)
    #     tril_mask = np.tril(np.ones((input_lengths, input_lengths), dtype=np.float32))
    #     attention_mask = (attention_mask-tril_mask)*np.finfo(np.float32).min
    #     return attention_mask
    # else:
    if dim4:
        total_len = kvcache_len + input_length
        attention_mask = np.ones((1, 1, input_length, total_len), dtype=np.float16)
        cache_mask = np.ones((1, 1, input_length, total_len), dtype=np.float16)
        cache_mask[:, :, :, cache_pos+1:-1] = 0
        attention_mask = (attention_mask-cache_mask)*np.finfo(np.float16).min
        return attention_mask
    else: 
        total_len = kvcache_len + input_length
        #attention_mask = np.ones((1, 1, 1, total_len), dtype=np.int64)
        cache_mask = np.ones((1, total_len), dtype=np.int64)
        cache_mask[:, cache_pos+1:-1] = 0
        return cache_mask

#print(f"position_ids input shape: {sessions['llm.onnx'].get_inputs()[1].shape}, type: {sessions['llm.onnx'].get_inputs()[1].type}")
def get_position_ids(position=0, image_start_pos=5, len_image_embeds=225):
    if position < image_start_pos:
        position_ids = np.array([[position]], dtype=np.int64)
    elif position >= image_start_pos and position < image_start_pos + len_image_embeds:
        position_ids = np.array([[image_start_pos]], dtype=np.int64)
        position_ids = np.repeat(position_ids[np.newaxis], 3, axis=0)
        ## 3d position
        position_ids[0, 0, 0] = image_start_pos
        position_ids[1, 0, 0] = (position - image_start_pos) // 15
        position_ids[2, 0, 0] = (position - image_start_pos) % 15
        return position_ids
    else:
        position_ids = np.array([[position-(len_image_embeds)+1]], dtype=np.int64)
    # convert from 1,1 to 3,1,1
    position_ids = np.repeat(position_ids[np.newaxis], 3, axis=0)
    return position_ids

def get_cache_postition(position):
    cache_position = np.array([position], dtype=np.int64)
    return cache_position

# attention_mask
cache_pos = -1
# past_key_values
# kv cache
past_key_values = np.zeros((num_layers, 2, 1, 2, kvcache_len, 128), dtype=datatype)
if PACT:
    past_key_values_1 = np.zeros((4, 2, 1, 2, kvcache_len, 128), dtype=datatype)
    past_key_values_2 = np.zeros((24, 2, 1, 2, kvcache_len//2, 128), dtype=datatype)
# cache_pos --> the position of the kv cache

# Generate tokens
start_time = time.time()
max_iter = 1024
query_states_before_rope_list = []
key_states_before_ropes_list = []
key_states_list = []
hidden_states_list = []
real_postition_ids = []
llm2_weights = None
pruned_length = -1

for i in range(max_iter):  # MAX_ITERATIONS
    
    position_ids= get_position_ids(position=i, image_start_pos=5, len_image_embeds=225)
    attention_mask = get_attention_mask(input_length=1, cache_pos=cache_pos)
    #cache_position = get_cache_postition(position=i)
    if i < pre_embeds_len:
        input_embed = np.expand_dims(embeddings[:, i, :], axis=0)
        
    else:
        input_embed = get_embeddings(token_id)
        #input_embed = np.expand_dims(input_embed, axis=0)
    
    # stage 1
    image_end_indice = 230
    if PACT and i < pre_embeds_len:
        hidden_states, new_kvcache, query_states_before_rope, key_states_before_rope = sessions['llm1.onnx'].run(
            outputs['llm1.onnx'],
            dict(zip(inputs['llm1.onnx'],
                [attention_mask, position_ids, past_key_values_1, input_embed]))
        )
        key_states = new_kvcache[3, 0]
        past_key_values_1[:, :, :, :, i:i+1, :] = new_kvcache[:, :, :, :, :, :]
        cache_pos += 1

        query_states_before_rope_list.append(query_states_before_rope)
        key_states_before_ropes_list.append(key_states_before_rope)
        key_states_list.append(key_states)
        hidden_states_list.append(hidden_states)
        real_postition_ids.append(position_ids)

        if i == pre_embeds_len-1:

            query_states_before_rope = np.concatenate(query_states_before_rope_list, axis=2)
            key_states_before_rope = np.concatenate(key_states_before_ropes_list, axis=2)
            key_states = np.concatenate(key_states_list, axis=2)
            hidden_states = np.concatenate(hidden_states_list, axis=1)
            real_postition_ids = np.concatenate(real_postition_ids, axis=2)


            image_hidden_states = hidden_states
            pruned_hidden_states, pruned_position_ids, weights, reduction= pact(image_hidden_states, query_states_before_rope, key_states_before_rope, key_states, None, image_mask, real_postition_ids)
            print(weights)
            print(pruned_position_ids)
            print(reduction)
            pruned_length = pruned_hidden_states.shape[1]
            
            llm2_weights = weights
            for j in range(pruned_length):

                attention_mask = get_attention_mask(input_length=1, cache_pos=j-1, kvcache_len=512, dim4=True)
                attention_mask[..., :j+1] = weights[:, :j+1].view(1, 1, 1, -1).cpu().numpy().astype(datatype)
                position_ids = pruned_position_ids[:, :, j:j+1].cpu().numpy()
                hidden_states = pruned_hidden_states[:, j:j+1, :].cpu().numpy().astype(datatype)
                
                logits, new_kvcache = sessions['llm2.onnx'].run(
                    outputs['llm2.onnx'],
                    dict(zip(inputs['llm2.onnx'],
                        [attention_mask, position_ids, past_key_values_2, hidden_states]))
                )

                past_key_values_2[:, :, :, :, j:j+1, :] = new_kvcache[:, :, :, :, :, :]

            token_id = np.argmax(logits)

            if token_id in [151643, 151645]:  # End tokens
                print(token_id, tokenizer.decode(token_id))
                break

            print(tokenizer.decode(token_id), end='', flush=True)
            

        # stage 2
    elif PACT and i >= pre_embeds_len:

        hidden_states, new_kvcache, query_states_before_rope, key_states_before_rope = sessions['llm1.onnx'].run(
            outputs['llm1.onnx'],
            dict(zip(inputs['llm1.onnx'],
                [attention_mask, position_ids, past_key_values_1, input_embed]))
        )
        past_key_values_1[:, :, :, :, i:i+1, :] = new_kvcache[:, :, :, :, :, :]

        new_index = pruned_length+i-pre_embeds_len
        attention_mask = get_attention_mask(input_length=1, cache_pos=new_index-1, kvcache_len=512, dim4=True)
        attention_mask[..., :j+1] = llm2_weights[:, :j+1].view(1, 1, 1, -1).cpu().numpy().astype(datatype)

        logits, new_kvcache = sessions['llm2.onnx'].run(
            outputs['llm2.onnx'],
            dict(zip(inputs['llm2.onnx'],
                [attention_mask, position_ids, past_key_values_2, hidden_states]))
        )
        
        token_id = np.argmax(logits)

        if token_id in [151643, 151645]:  # End tokens
            print(token_id, tokenizer.decode(token_id))
            break

        print(tokenizer.decode(token_id), end='', flush=True)
        past_key_values_2[:, :, :, :, new_index:new_index+1, :] = new_kvcache[:, :, :, :, :, :]
        cache_pos += 1
    ## normal inference
    else:
        logits, new_kvcache = sessions['llm.onnx'].run(
            outputs['llm.onnx'],
            dict(zip(inputs['llm.onnx'],
                [attention_mask, position_ids, past_key_values, input_embed]))
        )

        if i >= pre_embeds_len-1:
            token_id = np.argmax(logits)
            if token_id in [151643, 151645] and i >= pre_embeds_len:  # End tokens
                print(token_id, tokenizer.decode(token_id))
                break

            print(tokenizer.decode(token_id), end='', flush=True)
        # update kv cache
        past_key_values[:, :, :, :, i:i+1, :] = new_kvcache[:, :, :, :, :, :]
        cache_pos += 1


print(f"\nTotal time: {time.time() - start_time:.2f}s")