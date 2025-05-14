atc --framework=5 \
    --model="/home/chenl/project/ascend/llm-export/script/model/onnx/embedder.onnx" \
    --output="qwen2-vl-embedder" \
    --input_format=ND \
    --input_shape="input_ids:1,1" \
    --log=debug \
    --soc_version=Ascend310B1 \
    --precision_mode=must_keep_origin_dtype