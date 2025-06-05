# per-token absmax量化
def get(model_cfg,act_max):
	quant_cfg = {}
	#if model_cfg has num_hidden_layers
	if hasattr(model_cfg,"num_hidden_layers"):
		for i in range(model_cfg.num_hidden_layers):
			for name in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","qkv","proj","mlp"]:
				quant_cfg[str(i)+"."+name] = {"type":"W8X8"}
		quant_cfg["lm_head"] = {"type":"W8X8"}
	elif hasattr(model_cfg, "depth"):
		for i in range(model_cfg.depth):
			for name in ["qkv","proj","fc1","fc2"]:
				quant_cfg[str(i)+"."+name] = {"type":"W8X8"}
		quant_cfg["merger.0"] = {"type":"W8X8"}
		quant_cfg["merger.2"] = {"type":"W8X8"}
		
	return quant_cfg