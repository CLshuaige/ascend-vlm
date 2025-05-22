atc --framework=5 \
    --model="embedder.onnx" \
    --output="/root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/embedder" \
    --input_format=ND \
    --input_shape="input_ids:1,1" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype

# kvcache 1024 inputlen 1
atc --framework=5 \
    --model="fixed_llm.onnx" \
    --output="/root/Documents/model_dir/Ascend_llms/Qwen2VL/fixed_llm" \
    --input_format=ND \
    --input_shape="input_ids:1,1,1536;attention_mask:1,1,1,1025;position_ids:1,1;past_key_values:28,2,1,1024,2,128" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype

# source
atc --framework=5 \
    --model="llm_4dim_attn_sim.onnx" \
    --output="/root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/llm" \
    --input_format=ND \
    --input_shape="attention_mask:1,1,1,1025;position_ids:3,1,1;past_key_values:28,2,1,2,1024,128;input_embeds:1,1,1536" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype

#pact 1-4
atc --framework=5 \
    --model="llm_1_4.onnx" \
    --output="/root/Documents/model_dir/Ascend_llms/Qwen2VL/int8_pact/llm_1_4" \
    --input_format=ND \
    --input_shape="attention_mask:1,1,1,1025;position_ids:3,1,1;past_key_values:4,2,1,2,1024,128;input_embeds:1,1,1536" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype

# pact 5-28
atc --framework=5 \
    --model="llm_5_28.onnx" \
    --output="/root/Documents/model_dir/Ascend_llms/Qwen2VL/int8_pact/llm_5_28" \
    --input_format=ND \
    --input_shape="attention_mask:1,1,1,257;position_ids:3,1,1;past_key_values:24,2,1,2,256,128;hidden_states:1,1,1536" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype

atc --framework=5 \
    --model="llm.onnx" \
    --output="/root/Documents/model_dir/Ascend_llms/Qwen2VL/int8_1024_dy/llm" \
    --input_format=ND \
    --input_shape="attention_mask:1,1,1,1025;position_ids:3,1,1;past_key_values:28,2,1,2,1024,128;input_embeds:1,1,1536" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype



atc --framework=5 \
    --model="visual.onnx" \
    --output="/root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/visual" \
    --input_format=ND \
    --input_shape="pixel_values:1,900,1176" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype

atc --framework=5 \
    --model="visual_image_embeds_nocu.onnx" \
    --output="/root/Documents/model_dir/Ascend_llms/Qwen2VL/visual_image_embeds_nocu" \
    --input_format=ND \
    --input_shape="hidden_states:900,1280;embcos:900,80;embsin:900,80" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype


export MAX_COMPILE_CORE_NUMBER=1
export TE_PARALLEL_COMPILER=1