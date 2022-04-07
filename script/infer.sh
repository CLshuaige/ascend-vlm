KVSIZE=256
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


python ./inference/main.py \
    --model_type qwen2vl-2b \
    --vision_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/visual.om \
    --embedding_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/fp16_1024_1/embedder.om \
    --llm_model /root/Documents/model_dir/Ascend_llms/Qwen2VL/int8_1024_dy/llm_256.om \
    --hf-dir /root/Documents/model_dir/Qwen2-VL-2B-Instruct \
    --engine acl \
    --sampling greedy --sampling_value 10 --temperature 0.7 \
    --kvcache sliding-window \
    --cli \
    --kv_size ${KVSIZE} \
    --visual_path /root/Documents/project/qwenvl_infer/demo.jpeg