from config import InferenceConfig
from kvcache import KVCache
import numpy as np
from typing import List, Optional
import time
import sys
from engine import ACLModel,initResource

from utils_pact import pact

from tqdm import tqdm

class Session:
	def __init__(self,config:InferenceConfig) -> None:
		if config.pact:
			config.n_layer = config.n_layer_pact1
			self.kvCache1 = KVCache.create(config)
			config.n_layer = config.n_layer_pact2
			config.max_cache_size = int(config.max_cache_size * config.reduction_ratio_for_cache_size)
			self.kvCache2 = KVCache.create(config)
			self.kvCache = [self.kvCache1,self.kvCache2]
		else:
			self.kvCache = KVCache.create(config)
		self.max_len = config.max_input_len

	def run(self,input_ids:np.ndarray):
		pass
	
	@staticmethod
	def fromConfig(config:InferenceConfig) -> 'Session':
		if config.session_type == "onnx":
			return OnnxSession(config)
		elif config.session_type=='acl':
			if config.model_type == 'qwen2vl-2b':
				return AclQwenVLSession(config)
			elif config.model_type == 'qwen2vl-pact':
				return AclQwenVLPACTSession(config)
			else:
				return AclSession(config)
		else:
			return None
	
	def reset(self):
		if isinstance(self.kvCache, list):
			for kvCache in self.kvCache:
				kvCache.reset()
		else:
			self.kvCache.reset()

	def rollback(self,seq_len):
		if isinstance(self.kvCache, list):
			for kvCache in self.kvCache:
				kvCache.rollback(seq_len)
		else:
			self.kvCache.rollback(seq_len)

	def evict(self,space_need):
		self.kvCache.evict(space_need)
	
class OnnxSession(Session):
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		import onnxruntime
		options = onnxruntime.SessionOptions()
		self.llm_session = onnxruntime.InferenceSession(
            config.model,
            sess_options=options,
            providers=[
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

	def run(self,input_ids:np.ndarray):
		seq_len=input_ids.shape[-1]
		l,r,result = 0,self.max_len,None
		while l < seq_len:
			r = min(seq_len,r)
			cache,mask,pos_ids = self.kvCache.getInputs(r-l)
			result = self.llm_session.run(None,{
				"input_ids": input_ids[:,l:r],
				"attention_mask":mask,
				"past_key_values": cache,
				"position_ids": pos_ids,
			})
			# result:  [logits,key_values,attn_scores]
			self.kvCache.update(r-l,result[1],result[2])
			l , r = l+self.max_len , r + self.max_len
		return result

class AclSession(Session):
	context = None
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		from engine import ACLModel,initResource
		self.context = initResource(config.device)
		self.model = ACLModel(config.model,context=self.context,mode=config.acl_mode)
		self.input_ids = np.zeros((1,self.max_len),dtype=np.int64)
		if config.acl_mode == 'rc':
			self.input_ids,_,_,self.kvCache.kvCache = self.model.getInputs()

	def run(self,input_ids:np.ndarray):
		seq_len=input_ids.shape[-1]
		l,r,result = 0,self.max_len,None
		while l < seq_len:
			r = min(seq_len,r)
			self.input_ids[:,:r-l] = input_ids[:,l:r]
			cache,mask,pos_ids = self.kvCache.getInputs(self.max_len)
			result:List[np.ndarray] = self.model.inference([self.input_ids,mask,pos_ids,cache])
			# result:  [logits,key_values,attn_scores]
			self.kvCache.update(r-l,result[1],result[2])
			l , r = l+self.max_len , r + self.max_len
		return result
	
class AclQwenVLSession(Session):
	context = None
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		self.context = initResource(config.device)
		self.acl_mode = config.acl_mode
		# input: attention_mask:1,1,1,1025;position_ids:3,1,1;past_key_values:28,2,1,2,1024,128;input_embeds:1,1,1536
		if config.visual_path is not None:
			self.vision_model = ACLModel(config.vision_model,context=self.context,mode=config.acl_mode)
		self.llm_model = ACLModel(config.llm_model,context=self.context,mode=self.acl_mode)
		# input: pixel_values:1,900,1176
		self.embedding_model = ACLModel(config.embedding_model,context=self.context,mode=config.acl_mode)
		# input: input_ids:1,1
		self.image_pad_id = config.image_pad_id
		self.format = config.format

		self.input_ids = np.zeros((1,self.max_len),dtype=np.int64)
		self.input_embeds = np.zeros((1,self.max_len,config.hidden_dim),dtype=np.float16)
		# if self.acl_mode == 'rc':
		# 		_, _, self.kvCache.kvCache, self.input_embeds = self.llm_model.getInputs()


	def run(self, input_ids:np.ndarray,  image_mask, pixel_values=None,):
		seq_len=input_ids.shape[-1]
		l,r,result = 0,self.max_len,None
		
		if pixel_values is not None:

			pixel_values = np.expand_dims(pixel_values,axis=0)
			image_embeds = self.vision_model.inference([pixel_values])[0]
			image_embeds = image_embeds.reshape(1,image_embeds.shape[0],-1).astype(np.float16)
			image_start_pos = np.where(image_mask==True)[1][0]
			image_len = np.sum(image_mask)
			self.vision_model.unload()
		else:
			image_start_pos = -1
			image_len = 0

		pbar = None
		if seq_len > 1:
			pbar = tqdm(total=seq_len,desc='Inference')
		while l < seq_len:
			r = min(seq_len,r)
			self.input_ids[:,:r-l] = input_ids[:,l:r]
			cache,mask,pos_ids = self.kvCache.getInputsForVLM(self.max_len,image_mask)
			if l < image_start_pos:
				self.input_embeds[:,:r-l,:] = self.embedding_model.inference([self.input_ids])[0]
			elif l >= image_start_pos + image_len:
				self.input_embeds[:,:r-l,:] = self.embedding_model.inference([self.input_ids])[0]
			else:
				self.input_embeds[:,:r-l,:] = image_embeds[:,l-image_start_pos:r-image_start_pos,:]
				
			result:List[np.ndarray] = self.llm_model.inference([mask,pos_ids,cache,self.input_embeds])
			if pbar is not None:
				pbar.update(r-l)
			if self.format == 'huggingface-tensor':
				self.kvCache.update(r-l, result[1])
			l , r = l+self.max_len , r + self.max_len
		if pbar is not None:
			pbar.close()
		return result

class AclQwenVLPACTSession(Session):
	context = None
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		self.context = initResource(config.device)
		self.acl_mode = config.acl_mode

		self.vision_model = ACLModel(config.vision_model,context=self.context,mode=config.acl_mode)
		self.embedding_model = ACLModel(config.embedding_model,context=self.context,mode=config.acl_mode)
		
		llm_model_1 = ACLModel(config.llm_model.replace('llm.om','llm_1_4.om'),context=self.context,mode=self.acl_mode)
		llm_model_2 = ACLModel(config.llm_model.replace('llm.om','llm_5_28.om'),context=self.context,mode=self.acl_mode)
		self.llm_model = [llm_model_1, llm_model_2]

		self.image_pad_id = config.image_pad_id
		self.format = config.format

		self.input_ids = np.zeros((1,self.max_len),dtype=np.int64)
		self.input_embeds = np.zeros((1,self.max_len,config.hidden_dim),dtype=np.float16)
		self.hidden_states = np.zeros((1,self.max_len,config.hidden_dim),dtype=np.float16)

		# pact args
		self.pact_config_path = config.pact_config_path
		self.query_states_before_rope_list = []
		self.key_states_before_rope_list = []
		self.key_states_list = []
		self.hidden_states_list = []
		self.real_position_ids = []

		self.weights = None
		# if self.acl_mode == 'rc':
		# 		_, _, self.kvCache[0].kvCache, self.input_embeds = self.llm_model_1.getInputs()
	def unload_args(self):

		self.query_states_before_rope_list = []
		self.key_states_before_rope_list = []
		self.key_states_list = []
		self.hidden_states_list = []
		self.real_position_ids = []

	def run(self, input_ids:np.ndarray, image_mask, pixel_values=None,):
		seq_len=input_ids.shape[-1]
		
		use_pact = (pixel_values is not None) # 是否需要在改RUN中使用pact reduction
		if pixel_values is not None:
			pixel_values = np.expand_dims(pixel_values,axis=0)
			image_embeds = self.vision_model.inference([pixel_values])[0]
			image_embeds = image_embeds.reshape(1,image_embeds.shape[0],-1).astype(np.float16)
			image_start_pos = np.where(image_mask==True)[1][0]
			image_len = np.sum(image_mask)
			self.vision_model.unload()
		else:
			image_start_pos = -1
			image_len = 0

		hidden_states_for = None
		# llm_1_4
		l,r,result_1 = 0,self.max_len,None
		pbar = None
		if seq_len > 20:
			pbar = tqdm(total=seq_len,desc='Inference1')
		while l < seq_len:
			r = min(seq_len,r)
			self.input_ids[:,:r-l] = input_ids[:,l:r]
			cache,mask,pos_ids = self.kvCache[0].getInputsForVLM(self.max_len,image_mask)
			if l < image_start_pos:
				self.input_embeds[:,:r-l,:] = self.embedding_model.inference([self.input_ids])[0]
			elif l >= image_start_pos + image_len:
				self.input_embeds[:,:r-l,:] = self.embedding_model.inference([self.input_ids])[0]
			else:
				self.input_embeds[:,:r-l,:] = image_embeds[:,l-image_start_pos:r-image_start_pos,:]
			
			# hidden_states, new_kvcache, query_states_before_rope, key_states_before_rope
			hidden_states, new_kvcache, query_states_before_rope, key_states_before_rope = self.llm_model[0].inference([mask,pos_ids,cache,self.input_embeds])

			if use_pact:
				key_states = new_kvcache[3, 0]
				self.hidden_states_list.append(hidden_states.copy())
				self.query_states_before_rope_list.append(query_states_before_rope.copy())
				self.key_states_before_rope_list.append(key_states_before_rope.copy())
				self.key_states_list.append(key_states.copy())
				self.real_position_ids.append(pos_ids.copy())
			else:
				self.hidden_states_list.append(hidden_states.copy())

			if pbar is not None:
				pbar.update(r-l)
			if self.format == 'huggingface-tensor':
				self.kvCache[0].update(r-l, new_kvcache)
			l , r = l+self.max_len , r + self.max_len
		if pbar is not None:
			pbar.close()


		if use_pact:
			query_states_before_rope = np.concatenate(self.query_states_before_rope_list, axis=2)
			key_states_before_rope = np.concatenate(self.key_states_before_rope_list, axis=2)
			key_states = np.concatenate(self.key_states_list, axis=2)
			image_hidden_states = np.concatenate(self.hidden_states_list, axis=1)
			real_position_ids = np.concatenate(self.real_position_ids, axis=2)

			
			pruned_hidden_states, pruned_position_ids, self.weights, reduction= pact(image_hidden_states, query_states_before_rope, key_states_before_rope, key_states, None, image_mask[:, -seq_len:], real_position_ids, self.pact_config_path)
			print(f"PACT 保留率: {1.0*reduction[1]/reduction[0]:.2f}%")
			seq_len = pruned_hidden_states.shape[1]

		# llm_4_28
		l,r,result_2 = 0,self.max_len,None
		pbar = None
		if seq_len > 20:
			pbar = tqdm(total=seq_len,desc='Inference2')
		while l < seq_len:
			r = min(seq_len,r)
			cache,mask,pos_ids = self.kvCache[1].getInputsForVLM(self.max_len,image_mask)

			if use_pact:
				mask[..., :l+1] = self.weights[:, :l+1].view(1, 1, 1, -1).cpu().numpy().astype(np.float16)
				pos_ids = pruned_position_ids[:, :, l:l+1].cpu().numpy()
				self.hidden_states = pruned_hidden_states[:, l:l+1, :].cpu().numpy().astype(np.float16)
			else:
				if self.weights is not None:
					weights_end_pos = self.weights.shape[1]
					mask[..., :weights_end_pos] = self.weights[:, :weights_end_pos].view(1, 1, 1, -1).cpu().numpy().astype(np.float16)
				self.hidden_states = self.hidden_states_list[l]
				
			# print(f"pos_ids: {pos_ids[0]}")
			# print(f"KVcache[1], input_pos: {self.kvCache[1].input_pos}, p: {self.kvCache[1].p}, kv_size: {self.kvCache[1].kv_size}")
			result_2:List[np.ndarray] = self.llm_model[1].inference([mask,pos_ids,cache,self.hidden_states])

			if pbar is not None:
				pbar.update(r-l)
			if self.format == 'huggingface-tensor':
				self.kvCache[1].update(r-l, result_2[1])
			l , r = l+self.max_len , r + self.max_len
		if use_pact:
			self.kvCache[1].set_input_pos(self.kvCache[0].input_pos)
		if pbar is not None:
			pbar.close()
				
		self.unload_args()
		return result_2