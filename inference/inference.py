import numpy as np
import os
from typing import Any, Generator, List,Tuple,Dict
from threading import Lock
from session import Session
from config import InferenceConfig
from transformers import AutoTokenizer, AutoProcessor

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from qwen_vl_utils import process_vision_info
import base64
import re
from io import BytesIO
import time
import logging
import log_config

from npu_monitor import NPUMonitor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LlamaInterface:
    def __init__(self,config:InferenceConfig) -> None:
        self.max_length = config.max_length
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        self.state:dict[str,Any] = {"code":200,"isEnd":False,"message":""}        
        self.first=True
        self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5,"<|system|>":5}
        self.stop_words = ["<|user|>","<|assistant|>","<|system|>","[|AI|]","[|Human|]"]
        self.model_type = config.model_type
        self.last_output=""
        self.lock = Lock()
        self.reset()
        print("init success")

    def generate_cache(self,prompt:str):
        if len(prompt) == 0 :
            return
        input_ids = np.asarray(self.encode(prompt,add_bos_token=self.first),dtype=np.int64).reshape(1,-1)
        self.first=False
        logits = self.session.run(input_ids)[0]
        return self.sample_logits(logits[0][-1:],self.sampling_method,self.sampling_value,self.temperature),logits

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits.astype(np.float32)
            logits /= temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    
    def format_last_output(self):
        if len(self.last_output) == 0:
            return 
        text_format = self.apply_chat_template([{"role":"assistant","content":self.last_output}])
        self.generate_cache(text_format[len(self.last_output):])
        self.last_output = ""
    
    def predict(self, text):
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""        
        if text == "":
            return
        self.format_last_output()
        text = self.apply_chat_template([{"role":"user","content":text}])
        input_ids = self.encode(text,add_bos_token=self.first)
        input_ids = np.asarray(input_ids,dtype=np.int64).reshape(1,-1)
        self.first,ids_list = False,[]
        for i in range(self.max_length):
            logits = self.session.run(input_ids)[0]
            input_ids = self.sample_logits(logits[0][-1:], self.sampling_method, self.sampling_value, self.temperature)
            input_ids = input_ids.reshape(1, -1)
            if input_ids[0] == self.tokenizer.eos_token_id:
                self.session.rollback(1) 
                break
            ids_list.append(input_ids[0].item())
            text_out = self.tokenizer.decode(ids_list)
            stop_word = is_stop_word_or_prefix(text_out,self.stop_words)
            if stop_word != "":
                ids_list = ids_list[:-self.stop_mp[stop_word]]
                self.session.rollback(self.stop_mp[stop_word]) 
                break
            if i%3 == 0:
                with self.lock:
                    self.state['message']=text_out
        self.last_output = self.tokenizer.decode(ids_list)
        with self.lock:
            self.state['message'],self.state['isEnd'] = self.last_output,True
        return self.last_output

    def reset(self):
        self.first = True
        self.last_output = ""
        self.session.reset()
        self.generate_cache(self.apply_chat_template(self.prompt))
        
    def getState(self):
        with self.lock:
            return self.state.copy()

    def apply_chat_template(self,messages:List[Dict[str,str]]) -> str:
        text = ""
        if self.model_type == "llama-2-7b":
            for message in messages:
                if message["role"] == "user":
                    text += f'[|Human|]\n{message["content"]}\n[|AI|]'
                elif message["role"] == "system":
                    text += f'[|System|]\n{message["content"]}\n'
                else:
                    text += f'{message["content"]}\n'
        elif self.model_type == "tiny-llama":
            for message in messages:
                if message["role"] == "user":
                    text += f'<|user|>\n{message["content"]}</s>\n<|assistant|>'
                elif message["role"] == "system":
                    text += f'<|system|>\n{message["content"]}</s>\n'
                else:
                    text += f'{message["content"]}</s>\n'
        return text
    
    def encode(self,text,add_bos_token=False):
        self.tokenizer.add_bos_token = add_bos_token
        return self.tokenizer.encode(text)


class Qwen2VLInterface:
    def __init__(self,config:InferenceConfig) -> None:

        self.monitor = NPUMonitor(interval=0.5, log_file='./logs/npu_memory.log')
        self.monitor.start()

        self.max_length = config.max_length
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer)
        self.processor = AutoProcessor.from_pretrained(config.hf_model_dir, use_fast=True)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        self.system_prompt = config.system_prompt
        self.state:dict[str,Any] = {"code":200,"isEnd":False,"message":""}        
        self.first=True

        self.resized_height, self.resized_width = config.image_size, config.image_size
        ## TODO
        # self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5,"<|system|>":5}
        # self.stop_words = ["<|user|>","<|assistant|>","<|system|>","[|AI|]","[|Human|]"]
        self.model_type = config.model_type
        self.last_output=""

        self.image_mask = []
        self.image_pad_id = config.image_pad_id
        self.chat_history = "" # save the history of chat

        self.processor = AutoProcessor.from_pretrained(config.hf_model_dir, use_fast=True)
        self.is_token_by_token = config.is_token_by_token

        self.lock = Lock()
        self.reset()

        logging.info("init success")

    def generate_cache(self,prompt:str):
        if len(prompt) == 0 :
            return
        self.chat_history += prompt
        input_ids = np.asarray(self.encode(prompt,add_bos_token=self.first),dtype=np.int64).reshape(1,-1)
        image_mask = input_ids == self.image_pad_id
        if self.image_mask == []:
            self.image_mask = image_mask
        else:
            self.image_mask = np.concatenate((self.image_mask, image_mask), axis=1)
        self.first=False
        logits = self.session.run(input_ids, self.image_mask)[0]
        return self.sample_logits(logits[0][-1:],self.sampling_method,self.sampling_value,self.temperature),logits

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits.astype(np.float32)
            logits /= temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    def clear(self):
        import sys
        if isinstance(self.session.llm_model, List):
            for model in self.session.llm_model:
                model.unload()
        else:
            self.session.llm_model.unload()
        self.session.embedding_model.unload()
        self.monitor.stop()
        logging.debug(f"history: {self.chat_history}")
        logging.info("exit!")
        sys.exit(0)


    def format_last_output(self):
        if len(self.last_output) == 0:
            return 
        self.generate_cache('<|im_end|>\n')
        self.last_output = ""
    
    def predict(self, text, image=None):
        start_total = time.time()
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""        
        if text == "":
            return
        if text == "exit":
            self.clear()
        self.format_last_output()
        if self.first:
            text, image_inputs= self.apply_chat_template([{"role":"system","content":self.system_prompt}, {"role":"user","content":(text,image)}])
            self.first = False
        else:
            text, image_inputs = self.apply_chat_template([{"role":"user","content":(text,image)}])
        self.chat_history += text
        inputs = self.processor(text=text, images=image_inputs, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        if image is not None:
            pixel_values = inputs["pixel_values"]
            pixel_values = np.asarray(pixel_values,dtype=np.float16)
        else:
            pixel_values = None
        input_ids = np.asarray(input_ids,dtype=np.int64).reshape(1,-1)
        image_mask = input_ids == self.image_pad_id
        if self.image_mask == []:
            self.image_mask = image_mask
        else:
            self.image_mask = np.concatenate((self.image_mask, image_mask), axis=1)
        self.first,ids_list = False,[]
        first_token = True
        for i in range(self.max_length):
            logits = self.session.run(input_ids, self.image_mask, pixel_values=pixel_values)[0]
            pixel_values = None
            input_ids = self.sample_logits(logits[0][-1:], self.sampling_method, self.sampling_value, self.temperature)
            input_ids = input_ids.reshape(1, -1)
            if input_ids[0] == self.tokenizer.eos_token_id:
                #self.session.rollback(1) 
                break
            ids_list.append(input_ids[0].item())
            if first_token:
                time_first_token = time.time()
                logging.info(f"first token time: {time_first_token - start_total}")
                first_token = False
            text_out = self.tokenizer.decode(ids_list)
            # stop_word = is_stop_word_or_prefix(text_out,self.stop_words)
            # if stop_word != "":
            #     ids_list = ids_list[:-self.stop_mp[stop_word]]
            #     self.session.rollback(self.stop_mp[stop_word]) 
            #     break
            if i%3 == 0:
                with self.lock:
                    self.state['message']=text_out
            if self.is_token_by_token:
                print(self.tokenizer.decode(ids_list[-1]),end="",flush=True)
        if self.is_token_by_token:
            print('\n',end="",flush=True)
        self.last_output = self.tokenizer.decode(ids_list)
        with self.lock:
            self.state['message'],self.state['isEnd'] = self.last_output,True
        self.chat_history += self.last_output
        end_total = time.time()
        logging.info(f"total time: {end_total - start_total}, token/second: {len(ids_list)/(end_total - time_first_token)}")
        
        if self.is_token_by_token:
            return ''
        return self.last_output

    def reset(self):
        self.first = True
        self.last_output = ""
        self.session.reset()
        #self.generate_cache(self.apply_chat_template([{"role":"system","content":self.system_prompt}])[0])
        
    def getState(self):
        with self.lock:
            return self.state.copy()

    def apply_chat_template(self,messages:List[Dict[str,str]]) -> str:
        text = ""
        image_inputs = None
        vision_token = ""
        if self.model_type == "qwen2vl-2b" or self.model_type == "qwen2vl-pact":
            for message in messages:
                if message["role"] == "user":
                    text_content, image = message["content"]
                    if image is not None:
                        if isinstance(image,Image.Image):
                            
                            base64_image = image.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visual = {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}",
                                                    "resized_height": self.resized_height, "resized_width": self.resized_width}
                            content_payload = [processed_visual] + [{"type": "text", "text": text_content}]
                            vision_token = '<|vision_start|><|image_pad|><|vision_end|>'
                    else:
                        content_payload = [{"type": "text", "text": text_content}]
                    processed_message = {
                            "role": "user",
                            "content": content_payload,
                        }
                    # text = self.processor.apply_chat_template([processed_message], tokenize=False, add_generation_prompt=True)
                    # print(f"text after apply: {text}")
                    text += f'<|im_start|>user\n{vision_token}{text_content}<|im_end|>\n<|im_start|>assistant\n'
                    if image:
                        image_inputs, _ = process_vision_info([processed_message])
                elif message["role"] == "system":
                    text += f'<|im_start|>system\n{message["content"]}<|im_end|>\n'
                else:
                    text += f'{message["content"]}\n'
        elif self.model_type == "tiny-llama":
            for message in messages:
                if message["role"] == "user":
                    text += f'<|user|>\n{message["content"]}</s>\n<|assistant|>'
                elif message["role"] == "system":
                    text += f'<|system|>\n{message["content"]}</s>\n'
                else:
                    text += f'{message["content"]}</s>\n'
        return text, image_inputs
    
    def encode(self,text,add_bos_token=False):
        self.tokenizer.add_bos_token = add_bos_token
        return self.tokenizer.encode(text)

def is_stop_word_or_prefix(s: str, stop_words: list) -> int:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return stop_word
    return ""

class InternVLInterface:
    def __init__(self,config:InferenceConfig) -> None:

        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        self.monitor = NPUMonitor(interval=0.5, log_file='./logs/npu_memory.log')
        self.monitor.start()

        self.max_length = config.max_length
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=True)
        # self.processor = AutoProcessor.from_pretrained(config.hf_model_dir, use_fast=True)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        self.system_prompt = config.system_prompt
        self.state:dict[str,Any] = {"code":200,"isEnd":False,"message":""}        
        self.first=True

        self.resized_height, self.resized_width = config.image_size, config.image_size
        ## TODO
        # self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5,"<|system|>":5}
        # self.stop_words = ["<|user|>","<|assistant|>","<|system|>","[|AI|]","[|Human|]"]
        self.model_type = config.model_type
        self.last_output=""

        self.image_mask = []
        self.image_pad_id = 92546
        self.chat_history = "" # save the history of chat

        # self.processor = AutoProcessor.from_pretrained(config.hf_model_dir, use_fast=True)
        self.is_token_by_token = config.is_token_by_token

        self.lock = Lock()
        self.reset()

        logging.info("init success")

    def generate_cache(self,prompt:str):
        if len(prompt) == 0 :
            return
        self.chat_history += prompt
        input_ids = np.asarray(self.encode(prompt,add_bos_token=self.first),dtype=np.int64).reshape(1,-1)
        image_mask = input_ids == self.image_pad_id
        if self.image_mask == []:
            self.image_mask = image_mask
        else:
            self.image_mask = np.concatenate((self.image_mask, image_mask), axis=1)
        self.first=False
        logits = self.session.run(input_ids, self.image_mask)[0]
        return self.sample_logits(logits[0][-1:],self.sampling_method,self.sampling_value,self.temperature),logits

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits.astype(np.float32)
            logits /= temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    def clear(self):
        import sys
        if isinstance(self.session.llm_model, List):
            for model in self.session.llm_model:
                model.unload()
        else:
            self.session.llm_model.unload()
        self.session.embedding_model.unload()
        self.monitor.stop()
        logging.debug(f"history: {self.chat_history}")
        logging.info("exit!")
        sys.exit(0)


    def format_last_output(self):
        if len(self.last_output) == 0:
            return 
        self.generate_cache('<|im_end|>\n')
        self.last_output = ""
    # TODO
    def predict(self, text, image=None):
        start_total = time.time()
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""        
        if text == "":
            return
        if text == "exit":
            self.clear()
        self.format_last_output()
        if self.first:
            text, pixel_values= self.apply_chat_template([{"role":"system","content":self.system_prompt}, {"role":"user","content":(text,image)}])
            self.first = False
        else:
            text, pixel_values = self.apply_chat_template([{"role":"user","content":(text,image)}])
        self.chat_history += text
        input_ids = self.encode(text)
        input_ids = np.asarray(input_ids,dtype=np.int64).reshape(1,-1)
        image_mask = input_ids == self.image_pad_id
        if self.image_mask == []:
            self.image_mask = image_mask
        else:
            self.image_mask = np.concatenate((self.image_mask, image_mask), axis=1)
        self.first,ids_list = False,[]
        first_token = True
        for i in range(self.max_length):
            logits = self.session.run(input_ids, self.image_mask, pixel_values=pixel_values)[0]
            pixel_values = None
            input_ids = self.sample_logits(logits[0][-1:], self.sampling_method, self.sampling_value, self.temperature)
            input_ids = input_ids.reshape(1, -1)
            if input_ids[0] == 92542:
                #self.session.rollback(1) 
                break
            ids_list.append(input_ids[0].item())
            if first_token:
                time_first_token = time.time()
                logging.info(f"first token time: {time_first_token - start_total}")
                first_token = False
            text_out = self.tokenizer.decode(ids_list)
            # stop_word = is_stop_word_or_prefix(text_out,self.stop_words)
            # if stop_word != "":
            #     ids_list = ids_list[:-self.stop_mp[stop_word]]
            #     self.session.rollback(self.stop_mp[stop_word]) 
            #     break
            if i%3 == 0:
                with self.lock:
                    self.state['message']=text_out
            if self.is_token_by_token:
                print(self.tokenizer.decode(ids_list[-1]),end="",flush=True)
        if self.is_token_by_token:
            print('\n',end="",flush=True)
        self.last_output = self.tokenizer.decode(ids_list)
        with self.lock:
            self.state['message'],self.state['isEnd'] = self.last_output,True
        self.chat_history += self.last_output
        end_total = time.time()
        logging.info(f"total time: {end_total - start_total}, token/second: {len(ids_list)/(end_total - time_first_token)}")
        
        if self.is_token_by_token:
            return ''
        return self.last_output

    def reset(self):
        self.first = True
        self.last_output = ""
        self.session.reset()
        #self.generate_cache(self.apply_chat_template([{"role":"system","content":self.system_prompt}])[0])
        
    def getState(self):
        with self.lock:
            return self.state.copy()
    # TODO
    def apply_chat_template(self,messages:List[Dict[str,str]]) -> str:
        text = ""
        image_inputs = None
        pixel_values = None
        vision_token = ""
        if self.model_type == "qwen2vl-2b" or self.model_type == "qwen2vl-pact":
            for message in messages:
                if message["role"] == "user":
                    text_content, image = message["content"]
                    if image is not None:
                        if isinstance(image,Image.Image):
                            
                            base64_image = image.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visual = {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}",
                                                    "resized_height": self.resized_height, "resized_width": self.resized_width}
                            content_payload = [processed_visual] + [{"type": "text", "text": text_content}]
                            vision_token = '<|vision_start|><|image_pad|><|vision_end|>'
                    else:
                        content_payload = [{"type": "text", "text": text_content}]
                    processed_message = {
                            "role": "user",
                            "content": content_payload,
                        }
                    # text = self.processor.apply_chat_template([processed_message], tokenize=False, add_generation_prompt=True)
                    # print(f"text after apply: {text}")
                    text += f'<|im_start|>user\n{vision_token}{text_content}<|im_end|>\n<|im_start|>assistant\n'
                    if image:
                        image_inputs, _ = process_vision_info([processed_message])
                elif message["role"] == "system":
                    text += f'<|im_start|>system\n{message["content"]}<|im_end|>\n'
                else:
                    text += f'{message["content"]}\n'
        elif self.model_type == "internvl":
            for message in messages:
                if message["role"] == "user":
                    text_content, image = message["content"]
                    if image is not None:
                        if isinstance(image,Image.Image):
                            base64_image = image.convert("RGB")
                            pixel_values = self.load_image(base64_image).to(torch.float16).numpy()
                            text += f'<|im_start|>user\n<img>{"<IMG_CONTEXT>" * 256}</img>\n{text_content}<|im_end|>\n<|im_start|>assistant\n'
                    else:
                        text += f'<|im_start|>user\n{text_content}<|im_end|>\n<|im_start|>assistant\n'

                elif message["role"] == "system":
                    text += f'<|im_start|>system\n{message["content"]}<|im_end|>\n'
                else:
                    text += f'{message["content"]}\n'
        elif self.model_type == "tiny-llama":
            for message in messages:
                if message["role"] == "user":
                    text += f'<|user|>\n{message["content"]}</s>\n<|assistant|>'
                elif message["role"] == "system":
                    text += f'<|system|>\n{message["content"]}</s>\n'
                else:
                    text += f'{message["content"]}</s>\n'
        return text, pixel_values
    
    def encode(self,text,add_bos_token=False):
        self.tokenizer.add_bos_token = add_bos_token
        return self.tokenizer.encode(text)

    def build_transform(self,input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self,aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self,image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self,image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        image = image.resize((input_size, input_size))
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values