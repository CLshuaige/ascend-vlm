import numpy as np
import os
from typing import Any, Generator, List,Tuple,Dict
from threading import Lock
from session import Session
from config import InferenceConfig
from transformers import AutoTokenizer, AutoProcessor

from PIL import Image
from qwen_vl_utils import process_vision_info
import base64
import re
from io import BytesIO

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
        self.max_length = config.max_length
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer)
        self.processor = AutoProcessor.from_pretrained(config.hf_model_dir, use_fast=True)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        self.state:dict[str,Any] = {"code":200,"isEnd":False,"message":""}        
        self.first=True

        self.resized_height, self.resized_width = config.image_size, config.image_size
        ## TODO
        self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5,"<|system|>":5}
        self.stop_words = ["<|user|>","<|assistant|>","<|system|>","[|AI|]","[|Human|]"]
        self.model_type = config.model_type
        self.last_output=""
        self.lock = Lock()
        self.reset()

        self.image_mask = []
        self.image_pad_id = config.image_pad_id



        self.processor = AutoProcessor.from_pretrained(config.hf_model_dir, use_fast=True)

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

    def clear(self):
        import sys
        print("clear...")
        self.session.llm_model.unload()
        self.session.embedding_model.unload()
        print("exit!")
        sys.exit(0)

    def format_last_output(self):
        if len(self.last_output) == 0:
            return 
        text_format = self.apply_chat_template([{"role":"assistant","content":self.last_output}])
        self.generate_cache(text_format[len(self.last_output):], self.image_mask)
        self.last_output = ""
    
    def predict(self, text, image=None):
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""        
        if text == "":
            return
        if text == "exit":
            self.clear()
        #self.format_last_output()

        text, image_inputs= self.apply_chat_template([{"role":"user","content":(text,image)}])
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
        self.first,ids_list = False,[]
        for i in range(self.max_length):
            logits = self.session.run(input_ids, self.image_mask, pixel_values=pixel_values)[0]
            pixel_values = None
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
        #self.generate_cache(self.apply_chat_template(self.prompt))
        
    def getState(self):
        with self.lock:
            return self.state.copy()

    def apply_chat_template(self,messages:List[Dict[str,str]]) -> str:
        text = ""
        image_inputs = None
        if self.model_type == "qwen2vl-2b":
            for message in messages:
                if message["role"] == "user":
                    text, image = message["content"]
                    if image:
                        if isinstance(image,Image.Image):
                            
                            base64_image = image.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visual = {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}",
                                                    "resized_height": self.resized_height, "resized_width": self.resized_width}
                            content_payload = [processed_visual] + [{"type": "text", "text": text}]
                    else:
                        content_payload = [{"type": "text", "text": text}]
                    message = {
                            "role": "user",
                            "content": content_payload,
                        }
                    text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
                    if image:
                        image_inputs, _ = process_vision_info([message])
                elif message["role"] == "system":
                    text += f'system\n{message["content"]}\n'
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