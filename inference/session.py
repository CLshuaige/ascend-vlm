from config import InferenceConfig
from kvcache import KVCache
import numpy as np
from typing import List, Optional
import time
import sys
from engine import ACLModel,initResource

from tqdm import tqdm

class Session:
	def __init__(self,config:InferenceConfig) -> None:
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
			else:
				return AclSession(config)
		else:
			return None
	
	def reset(self):
		self.kvCache.reset()

	def rollback(self,seq_len):
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

