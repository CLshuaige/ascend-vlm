import argparse
import importlib
import torch
import os
#from transformers import LlamaForCausalLM, LlamaTokenizer
import shutil
import transformers



shutil.copy("/home/chenl/project/ascend/ascend-llm/export_llama/modeling_qwen2_vl_m.py", "/home/chenl/miniconda3/envs/ascend-llm/lib/python3.9/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py")
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
#from qwen_vl_utils import process_vision_info

# device = cuda 0
device = "cuda:1"
opeset_version = 15

PACT = False
Layer1_4 = True
Layer5_28 = False

export_llm = True
export_embedder = False
export_visual = False

def export_onnx(base_model,out_path,quant_cfg_path,act_path):
    #= LlamaTokenizer.from_pretrained(base_model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(base_model)
    model_cfg=model.model.config
    llm_model = model
    spec = importlib.util.spec_from_file_location("quant_cfg_module", quant_cfg_path)
    quant_cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant_cfg_module)
    quantize_cfg = quant_cfg_module.get(model_cfg,act_path)
    from quantize import quantize
    quantize(llm_model,quantize_cfg)
    
    input_names = ["attention_mask", "position_ids","past_key_values", "input_embeds",
                   ]
    if PACT and Layer5_28:
        input_names = ["attention_mask", "position_ids","past_key_values", "hidden_states",]
    output_names = ["logits","out_key_values"]
    if PACT and Layer1_4:
        output_names = ["hidden_states","out_key_values", "query_states", "key_states"]
    dynamic_axes = {
        "attention_mask": { 0: "batch_size",2:"seq_length",3:"total_len" },
        "position_ids": { 1: "batch_size", 2: "seq_length" },
        "past_key_values": { 2: "batch_size", 4: "kv_len" },
        "input_embeds": { 0: "batch_size", 1: "seq_length" },
    }
    
    batch_size,seq_len,kv_len=1,1,1024
    all_len = seq_len + kv_len
    n_layers,n_heads,hidden_size=model_cfg.num_hidden_layers,model_cfg.num_key_value_heads,model_cfg.hidden_size

    head_dim = int(model_cfg.hidden_size / model_cfg.num_attention_heads)

    # vision parmas
    image_size = [420, 420]
    patch_size = [14, 14]
    patch_num_h = round(image_size[0]/patch_size[0])
    patch_num_w = round(image_size[1]/patch_size[1])
    patch_num = patch_num_w * patch_num_h
    value_num = patch_size[0] * patch_size[1] * 6

    image_token_num = patch_num // 4

    # get id of token 151655

    input_ids = torch.zeros((batch_size,seq_len)).long().to(device) # batch_size, new_sequence_length
    input_ids[:,:image_token_num] = 151655 # token 151655
    #attention_mask = torch.zeros((batch_size, all_len)).to(torch.int64).to(device) # batch_size, all_sequence_length
    attention_mask = torch.zeros((batch_size, 1, seq_len, all_len)).to(torch.float16).to(device) # batch_size, all_sequence_length
    attention_mask[:, -1] = 1
    position_ids = torch.zeros((3, batch_size,seq_len)).long().to(device) # batch_size, new_sequence_length
    # past_keys = torch.rand((batch_size,  n_heads,kv_len, head_dim),dtype=torch.float16).to("cuda")
    # past_values = torch.rand((batch_size,n_heads, kv_len, head_dim),dtype=torch.float16).to("cuda")
    # past_key_values = tuple([(past_keys,past_values)] * n_layers)
    past_key_values = torch.rand((n_layers, 2, batch_size, n_heads, kv_len, head_dim),dtype=torch.float16).to(device)
    if PACT and Layer1_4:
        past_key_values = torch.rand((4, 2, batch_size, n_heads, kv_len, head_dim),dtype=torch.float16).to(device)
    if PACT and Layer5_28:
        past_key_values = torch.rand((24, 2, batch_size, n_heads, kv_len, head_dim),dtype=torch.float16).to(device)

    inputs_embeds = llm_model.model.embed_tokens(input_ids).to(torch.float16)
    
    pixel_values = torch.rand((batch_size, patch_num, value_num),dtype=torch.float16).to(device)
    print("pixel_values shape:", pixel_values.shape)
    image_grid_thw = torch.tensor([[1, patch_num_h, patch_num_w]]).long().to(device)
    cache_position = torch.zeros((seq_len)).long().to(device)
    # print shapes of inputs
    print("input_ids shape:", input_ids.shape)
    print("attention_mask shape:", attention_mask.shape)
    print("position_ids shape:", position_ids.shape)
    #print("past_key_values shape:", past_key_values[0].shape)
    print("pixel_values shape:", pixel_values.shape)
    print("image_grid_thw shape:", image_grid_thw.shape)
    print("cache_position:", cache_position.shape)

    input_args = (
        None,#input_ids
        attention_mask,#attention_mask
        position_ids,#position_ids
        past_key_values,
        inputs_embeds,
    )

    llm_model.eval()
    if export_llm:
        torch.onnx.export(
            llm_model,
            f=out_path,
            args=input_args,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opeset_version,
            export_params=True,
        )

    if export_embedder:
        model = llm_model.model.embed_tokens
        out_path = out_path.replace("llm.onnx","embedder.onnx")
        input_args = (
            input_ids,
        )
        input_names = ["input_ids"]
        output_names = ["embeddings"]
        dynamic_axes = {
            "input_ids": { 0: "batch_size", 1: "seq_length" },
            "embeddings": { 0: "batch_size", 1: "seq_length" },
        }
        torch.onnx.export(
            model,
            f=out_path,
            args=input_args,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opeset_version,
            export_params=True,
        )

    if export_visual:
        model = llm_model.visual
        quantize(model,quantize_cfg)
        out_path = out_path.replace("llm.onnx","visual.onnx")
        input_args = (
            pixel_values,
        )
        input_names = ["pixel_values"]
        output_names = ["embeddings"]
        dynamic_axes = {
            "pixel_values": { 0: "batch_size", 1: "patch_num", 2: "value_num" },
        }
        torch.onnx.export(
            model,
            f=out_path,
            args=input_args,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opeset_version,
            export_params=True,
        )





if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m",
        type=str, 
        default="./model/TinyLlama-1.1B-Chat-v1.0", 
        help="transformers model"
    )
    parser.add_argument(
        "--output","-o",
        type=str,
        default="./model/export_out/tiny-llama.onnx",
        help="where to save onnx model",
    )
    parser.add_argument(
        "--act-path","-a",
        type=str,
        default="./act_scales/llama-2-7b.pt",
        help="path to act_scales",
    )
    parser.add_argument(
        "--quant","-q",
        type=str,
        default="./config/w8x8.py",
        help="path to quant config",
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="test onnx model",
    )

    args = parser.parse_args()
    if not args.test:
        export_onnx(args.model,args.output,args.quant,args.act_path)
