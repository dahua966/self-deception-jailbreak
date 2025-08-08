#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import json
import os
import re
import time
from abc import ABC
from ast import literal_eval
from datetime import datetime
from typing import Union, List, Dict
import argparse
from pprint import pprint
from tqdm import tqdm
from openai import OpenAI
from termcolor import cprint, colored
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import *


def jsonl_2_list(file_name, key=None):
    if not os.path.exists(file_name):
        return []
    if file_name.endswith(".jsonl"):
        if key:
            return [json.loads(line).get(key) for line in open(file_name, encoding="utf-8").readlines()]
        else:
            return [json.loads(line) for line in open(file_name, encoding="utf-8").readlines()]
    else:
        print(f"[!]File extension not supported.")
        return []


def save_with_indent(obj: Union[dict, list], filename: str, indent=True, append=False, slient=False):
    """Dump json object in indent or not"""
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    indent_file = filename.replace(".json", ".indent.json")
    fp_mode = "a" if append else "w"
    print(f"[*]File operation mode is \"{fp_mode}\"")

    if filename.endswith(".json"):
        indent_file = filename  # json file dumps in indent
        print(f"[*]Dumping json in indent mode (Default).")
    elif filename.endswith(".jsonl") and indent != False:
        assert type(obj) is list
        with open(filename, fp_mode, encoding="utf-8") as f:
            [f.write(f"{json.dumps(res, ensure_ascii=False)}\n") for res in obj]
    elif filename.endswith(".txt"):
        assert type(obj) is list
        with open(filename, fp_mode, encoding="utf-8") as f:
            [f.write(f"{res}\n") for res in obj]

    if indent:
        if indent_file.endswith(".json"):
            with open(indent_file, fp_mode, encoding="utf-8") as f:
                f.write(f"{json.dumps(obj, indent=4, ensure_ascii=False)}\n")
        elif indent_file.endswith(".jsonl"):
            with open(indent_file, fp_mode, encoding="utf-8") as f:
                [f.write(f"{json.dumps(res, indent=4, ensure_ascii=False)}\n") for res in obj]

        if indent == "only" and indent_file != filename:
            os.remove(filename)
            cprint(f"[+] Saved in {indent_file}", "green") if not slient else None
        else:
            cprint(f"[+] Saved in {filename} and {indent_file}", "green") if not slient else None
    else:
        cprint(f"[+] Saved in {filename}", "green") if not slient else None


class LLM(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.model_name = None
        self.device = "cuda"
        self.conversation = []
        self.tools = {}
        self.prompts = []
        self.convs = []

    def add_conv_node(self, conv: Union[Dict[str, str], List[Dict[str, str]]]):
        """Add a node to conversation list --> generate()"""
        if isinstance(conv, dict):
            self.conversation.append(conv)
        elif isinstance(conv, list):
            self.conversation.extend(conv)

    def clear_conv(self):
        self.conversation = []

    def set_conv(self, conv: List[Dict[str, str]]):
        """Set conversation for single generation --> generate()"""
        self.conversation = conv

    def set_tools(self, tools: List[Dict]):
        self.tools = tools

    def add_raw_prompt(self, prompt: str):
        """Add prompt for batch generation --> generate_batch()"""
        self.prompts.append(prompt)

    def add_conv_prompt(self, conversation: List[Dict[str, str]], thinking=True):
        """Add a completed conversation, turn into prompt str --> local model generate_batch()"""
        self.prompts.append(self.conv2prompt(conversation, thinking=thinking))

    def add_conv(self, conversation: List[Dict[str, str]]):
        """Add a completed conversation --> remote model generate_batch()"""
        self.convs.append(conversation)

    def set_prompts(self, prompts):
        self.prompts = prompts

    def clear_prompts(self):
        self.prompts = []

    def generate(self):
        pass

    @staticmethod
    def sys_prompt(prompt):
        return [{"role": "system", "content": prompt}]

    @staticmethod
    def user_prompt(prompt):
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def assistant_prompt(prompt):
        return [{"role": "assistant", "content": prompt}]

    @staticmethod
    def conv_prompt(prompt, response):
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]

    def conv2prompt(self, conversation: List[Dict], verbose=False, thinking=True) -> str:
        """将对话列表转化为prompt字符串 考虑模型没有自带对话模板的情况"""
        if self.tokenizer.chat_template is None:
            from fastchat.model import get_conversation_template
            model_name = self.model_name.lower()
            if "llama-3.1" in model_name:
                model_name = "llama-3-"  # patch for fastchat
            elif "glm-4" in model_name:
                model_name = "chatglm3"
            conv_template = get_conversation_template(model_name)
            if conv_template.name == "one_shot":
                cprint(f"[!]Warning: Conversation template is {conv_template.name}!", "yellow")
            cprint(f"\n[*]Conversation template is {conv_template.name}", "blue") if verbose else None

            for msg in conversation:
                if msg["role"] == "system":
                    conv_template.set_system_message(msg["content"])
                elif msg["role"] == "user":
                    conv_template.append_message(conv_template.roles[0], msg["content"])
                elif msg["role"] == "assistant":
                    conv_template.append_message(conv_template.roles[1], msg["content"])

            if conv_template.messages[-1][0] == conv_template.roles[0]:
                conv_template.append_message(conv_template.roles[1], "")
            else:
                raise Exception(f"Conversation's last role is not user.\n{conv_template.messages}\n")

            prompt = conv_template.get_prompt()
        else:
            if not thinking and self.model_name.lower().startswith("qwen3"):
                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            # if isinstance(prompt, list) and len(prompt):
            #     prompt = prompt[0]

        if verbose:
            print(f"{colored('Structured prompt:', 'blue', attrs=['bold'])}:\n{prompt}\n")

        return prompt


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='auto',
                 ):
        super().__init__()
        self.model_path = model_path
        self.model_name = model_path.split(os.sep)[-1]
        self.device = device
        # cprint(f"[*]Loading LLM model: {self.model_name} @ {self.device}\n", "blue")
        s_time = time.time()
        self.model, self.tokenizer = self.create_model()
        cprint(f"[*]Loading LLM model({self.model_name} @ {self.device}) in {time.time() - s_time:.2f} seconds\n",
               "green")

    # @torch.inference_mode()
    def create_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            low_cpu_mem_usage=True,
            # torch_dtype=dtype,
            trust_remote_code=True,
        )
        cprint("[*]Using FP16 and normal attention implementation", "green")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if model.generation_config.pad_token_id is None:
            cprint(
                f"[*]model.generation_config.pad_token_id is None, set to tokenizer.pad_token_id({tokenizer.pad_token_id})",
                "yellow")
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        fix_tokenizer_pad(tokenizer, self.model_name)

        return model, tokenizer

    def generate(self, temperature=1.0,
                 max_tokens=512,
                 verbose=False,
                 thinking=False, **kwargs) -> str:
        """Generate from conversation"""
        if self.conversation:
            prompt = self.conv2prompt(self.conversation, verbose, thinking=thinking)
        elif self.prompts:
            prompt = self.prompts[0]
            print(f"[*]Generate response by prompt: {prompt}\n") if verbose else None
        else:
            cprint(f"[-]Empty conversation or prompt, return None", "red")
            return None

        if isinstance(prompt[0], str):
            inputs = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.model.device)
        else:
            cprint(f"[!]Prompts is not str, return None", "red")
            return None

        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens, **kwargs
            )  # repetition_penalty=1.0
        else:
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_tokens, **kwargs
            )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0, len(inputs['input_ids'][0]):]
        self.clear_prompts()

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )


def fix_tokenizer_pad(tokenizer, model_name):
    """设置填充参数"""
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
        print(f"[*]tokenizer.padding_side is None, set to left\n")
    else:
        print(f"[*]tokenizer.padding_side default: {tokenizer.padding_side}")
        if tokenizer.padding_side != "left" and "qwen" in model_name.lower():
            cprint(f"[!]Force set tokenizer.padding_side to left.\n", "yellow")
            tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        cprint(f"[*]tokenizer.pad_token is None, set to eos_token ({tokenizer.eos_token})\n", "yellow")
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        cprint(f"[*]tokenizer.pad_token_id is None, set to eos_token_id ({tokenizer.eos_token_id})\n", "yellow")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


class OpenAILLM(LLM):
    def __init__(self, model_name, base_url, key, repeat_num=5):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self.key = key
        self.repeat_num = repeat_num
        self.client = OpenAI(base_url=self.base_url, api_key=self.key)

    def generate(self, **kwargs) -> str:
        for msg in self.conversation:
            if not msg["role"] in ["system", "user", "assistant"]:
                raise Exception(f"[-]Error role in conversation: \n{self.conversation}")

        response = "[!]Error:"
        for i in range(1, self.repeat_num + 1):
            try:
                chat = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation,
                    timeout=90,
                    **kwargs
                )

                if chat.choices:
                    if chat.choices[0].message.tool_calls:
                        response = chat.choices[0].message.tool_calls[0].function.arguments
                    else:
                        response = chat.choices[0].message.content
                    if response is not None:
                        break
                elif chat.error:
                    eMsg = chat.error.message
                    raise Exception(eMsg)
                else:
                    print("[!]Error and Choice are both None")
                    print(f"[*]Chat:\n{chat}\n")

            except KeyboardInterrupt:
                print("[-]KeyboardInterrupt, exiting...")
                exit()

            except Exception as e:
                err_msg = str(e)[:500]
                print(
                    f"[!]在第{i}次请求 `{self.model_name}` 时报错\n"
                    f"[-]报错信息 ({datetime.now().strftime('%Y/%m/%d %H:%M:%S')}): \n{err_msg}\n"
                    f"[*]Sleep {10*i}s")
                # print_exc()
                if i < self.repeat_num:
                    time.sleep(10*i)
                else:
                    response += f"\n{err_msg}"
        return response


def llm_adapter(model_name, **kwargs) -> Union[LocalLLM, LLM]:
    """kwargs contains device / base_url / key / repeat_num"""
    model_path = None
    if model_name in MODEL_PATH.keys():
        model_path = MODEL_PATH[model_name]
    elif os.path.exists(model_name):
        model_path = model_name

    if model_path is not None:
        cprint(f"[*]Loading LLM model ({model_name})...", "blue")
        return LocalLLM(model_path, **kwargs)
    elif kwargs.get('device'):
        kwargs.pop("device")

    cprint(f"[*]Loading LLM API({model_name})...\n", "blue")
    if kwargs.get("base_url") is None:
        kwargs["base_url"] = API_HOST
    if kwargs.get("key") is None:
        kwargs["key"] = DIRECT_KEY

    return OpenAILLM(model_name, **kwargs)


#从文本中提取虚拟场景题目或者反派名字
def extract_keyword(content) -> str:
    res = re.findall("```([^`]+)", content)
    # print(f"res: {res}")
    if res:
        #取``` ```包裹的为题目
        keyword = res[0].strip()
        keyword = re.sub(r'[\"`*#]', '', keyword)
        if len(keyword) > 80:
            #反引号可能把整个文本都包含进来了，取第一行为题目
            split_keyword = keyword.split("\n")[0].strip()
            if len(split_keyword) < 80:
                keyword = re.sub(r'[\"`*#]', '', split_keyword)
            else:
                keyword = ""
    else:
        #取双引号包含的为题目
        res = re.findall("\"([^\"]+)\"", content)
        if res:
            #取引号包裹的为题目
            keyword = res[0].strip()
            if len(keyword) < 80:
                keyword = re.sub(r'[\"`*#]', '', keyword)
            else:
                keyword = ""
        else:
            #取第一行为题目
            keyword = content.split("\n")[0].strip()
            if len(keyword) < 80:
                keyword = re.sub(r'[\"`*#]', '', keyword)
            else:
                keyword = ""
    # 取消"xx:"前缀
    keyword = re.sub(f"[\w\s]+:\s*", "", keyword, flags=re.IGNORECASE)
    return keyword


def json_request(client: LLM, prompt:[str, list], repeat_num=3, verbose=True, **kwargs) -> dict:
    """请求LLM，返回Json格式数据"""
    assert client is not None
    # print(f"[*]Prompt:\n{prompt}") if verbose else None
    if isinstance(prompt, str):
        client.set_conv(LLM.user_prompt(prompt))
    elif isinstance(prompt, list):
        client.set_conv(prompt)
    else:
        cprint(f"[-]Json Request 提示词格式错误，应该为 [str, list]", "red")
        return None

    for i in range(repeat_num):
        final_res = {}
        results = client.generate(**kwargs)
        # print(f"[*]Raw output from {client.model_name}:\n{results}\n{'-'*99}")
        if results is None:
            return {}

        think_content = re.findall(r"<think>\n(.*)\n</think>", results)  # ds输出格式
        if len(think_content):
            final_res["think"] = think_content[0]

        json_contents = re.findall(r"```json([\s\S]*)```$", results, re.I)
        if len(json_contents):
            strip_content = json_contents[-1]
        else:
            strip_content = results[results.find("{"): results.rfind("}") + 1]

        try:
            json_res = json.loads(strip_content)
        except Exception as e:
            # print(f"[*] Failed to load JSON: {e}") if verbose else None
            json_res = None

        if not json_res:
            try:
                json_res = literal_eval(strip_content)
            except Exception as e:
                print(f"[-] Error in parsing response.\nRaw content:\n{'+' * 66}\n{results}\n{'+' * 66}\n") if verbose else None
                continue

        if not isinstance(json_res, dict):
            cprint(f"[*] Failed to parse JSON: {type(json_res)}", "red") if verbose else None
            continue
        else:
            final_res = json_res

        if None not in list(final_res.values()):
            break

    return final_res


def getMd5(text, trunc = 32):
    md5 = hashlib.md5()
    md5.update(text.encode("utf-8"))
    return md5.hexdigest()[:trunc]


def print_freedom():
    print(r'''
___________                         .___              
\_   _____/______   ____   ____   __| _/____   _____  
 |    __) \_  __ \_/ __ \_/ __ \ / __ |/  _ \ /     \ 
 |     \   |  | \/\  ___/\  ___// /_/ (  <_> )  Y Y  \
 \___  /   |__|    \___  >\___  >____ |\____/|__|_|  /
     \/                \/     \/     \/            \/  freedom!
    ''')