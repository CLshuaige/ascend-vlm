export PACT=False
export Layer1_4=False
export Layer5_28=False

export export_llm=False
export export_embedder=False
export export_visual=True


# python ../export_llama/export_qwen2_vl.py \
# 	--model /home/chenl/weights/hf-models/Qwen2-VL-2B-Instruct \
# 	--output /home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/fp16_1024_4dim_attention/llm/llm.onnx \
# 	#--quant /home/chenl/project/ascend/ascend-llm/export_llama/config/w8x8.py

#int8
python ../export_llama/export_qwen2_vl.py \
	--model /home/chenl/weights/hf-models/Qwen2-VL-2B-Instruct \
	--output /home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/int8_1024/visual/llm.onnx \
	--quant /home/chenl/project/ascend/ascend-llm/export_llama/config/w8x8.py
# PACT 1-4
# python ../export_llama/export_qwen2_vl.py \
# 	--model /home/chenl/weights/hf-models/Qwen2-VL-2B-Instruct \
# 	--output /home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/int8_pact/llm_1_4.onnx \
# 	--quant /home/chenl/project/ascend/ascend-llm/export_llama/config/w8x8.py

# python ../export_llama/export_qwen2_vl.py \
# 	--model /home/chenl/weights/hf-models/Qwen2-VL-2B-Instruct \
# 	--output /home/chenl/weights/export-models/Qwen2-VL-2B-Instruct/onnx_model/int8_pact/visual/visual_dy.onnx \
# 	--quant /home/chenl/project/ascend/ascend-llm/export_llama/config/w8x8.py