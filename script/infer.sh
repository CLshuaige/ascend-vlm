KVSIZE=1024
# python ./inference/main.py \
#     --model /root/Documents/model_dir/Ascend_llms/TinyLlama-1.1B-Chat-v1.0/tiny-llama-seq-1-key-256-int8.om \
#     --hf-dir /root/Documents/model_dir/TinyLlama-1.1B-Chat-v1.0 \
#     --engine acl \
#     --sampling greedy --sampling_value 10 --temperature 0.7 \
#     --cli \
#     --kv_size ${KVSIZE}
# python main.py \
#     --model /root/Documents/model_dir/Ascend_llms/Qwen2VL/ \
#     --hf-dir /root/Documents/model_dir/Qwen2-VL-2B-Instruct \
#     --engine acl \
#     --sampling top_k --sampling_value 10 --temperature 0.7 \
#     --cli \
#     --kv_size ${KVSIZE}


# python ./inference/main.py \
#     --model_type qwen2vl-2b\
#     --vision_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/visual.om \
#     --embedding_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/embedder.om \
#     --llm_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/int8_1024_dy/llm.om \
#     --hf-dir /root/Documents/model_dir/Qwen2-VL-2B-Instruct \
#     --engine acl \
#     --sampling greedy --sampling_value 10 --temperature 0.7 \
#     --kvcache basic \
#     --cli \
#     --kv_size ${KVSIZE} \
#     --visual_path /root/Documents/project/qwenvl_infer/demo.jpeg \
#     #--pact_config_path /root/Documents/project/ascend-vlm/inference/pact_configs.json

python ./inference/main.py \
    --model_type qwen2vl-pact\
    --vision_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/visual.om \
    --embedding_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/embedder.om \
    --llm_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/int8_pact/llm.om \
    --hf-dir /root/Documents/model_dir/Qwen2-VL-2B-Instruct \
    --engine acl \
    --sampling greedy --sampling_value 10 --temperature 0.7 \
    --kvcache sliding-window \
    --cli \
    --kv_size ${KVSIZE} \
    --visual_path /root/Documents/project/qwenvl_infer/demo.jpeg \
    --pact_config_path /root/Documents/project/ascend-vlm/inference/pact_configs.json \
    --tokenbytoken True \

# python ./inference/main.py \
#     --model_type qwen2vl-2b\
#     --vision_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/visual.om \
#     --embedding_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/embedder.om \
#     --llm_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/llm.om \
#     --hf-dir /root/Documents/model_dir/Qwen2-VL-2B-Instruct \
#     --engine acl \
#     --sampling greedy --sampling_value 10 --temperature 0.7 \
#     --kvcache basic \
#     --cli \
#     --kv_size ${KVSIZE} \
#     --visual_path /root/Documents/project/qwenvl_infer/demo.jpeg \
#     --tokenbytoken True \
#     #--pact_config_path /root/Documents/project/ascend-vlm/inference/pact_configs.json