python ../export_llama/export_llama.py \
	--model /home/chenl/weights/hf-models/Llama-2-7B-hf \
	--output /home/chenl/weights/export-models/Llama-2-7B-hf/llama2_7B_16_1024.onnx \
	--act-path /home/chenl/project/ascend/ascend-llm/export_llama/act_scales/llama-2-7b.pt \
	--quant /home/chenl/project/ascend/ascend-llm/export_llama/config/sd.py