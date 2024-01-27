import os
import pickle
import pandas as pd 
import numpy as np 
from langchain.prompts import PromptTemplate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

from langchain.output_parsers import ResponseSchema

def get_text_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def get_llama_template(temp, sys_prompt):
    '''Returns a properly formatted Llama template'''
    B_INST, E_INST = "[INST]\n\n", "\n[/INST]"
    B_SYS, E_SYS = "<SYS>\n", "<\SYS>\n"
    DEFAULT_SYSTEM_PROMPT = B_SYS + sys_prompt + E_SYS
    return B_INST + DEFAULT_SYSTEM_PROMPT + temp + E_INST

def get_llama_prompt(temp, sys_prompt, input_vars):
    '''Returns a properly formatted Llama prompt'''

    template = get_llama_template(temp, sys_prompt)
    prompt = PromptTemplate(template=template, input_variables=input_vars)

    return prompt

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def set_visible_gpus(visible):
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

def get_pipe(model_name, **kwargs):
    '''
    Please pass the following parameters: do_sample, output_scores, top_k, 
    return_dict, num_return_sequences, max_new_tokens
    '''

    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                            device_map='auto',
                                            torch_dtype=torch.float16,
                                            load_in_4bit=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.float16)

    # params = do_sample, output_scores, top_k, return_dict, num_return_sequences, max_new_tokens
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    eos_token_id=tokenizer.eos_token_id,
                    **kwargs
                    )

    return pipe

def get_llama_pipe(size, **kwargs):
    '''
    Please pass the following parameters: do_sample, output_scores, top_k, 
    return_dict, num_return_sequences, max_new_tokens
    '''

    if size == 7:
        return get_pipe('meta-llama/Llama-2-7b-chat-hf', **kwargs)
    elif size == 13:
        return get_pipe('meta-llama/Llama-2-13b-chat-hf', **kwargs)
    elif size == 70:
        return get_pipe('meta-llama/Llama-2-70b-chat-hf', **kwargs)

def get_llama_tokenizer(size, **kwargs):

    if size == 7:
        return get_tokenizer('meta-llama/Llama-2-7b-chat-hf')
    elif size == 13:
        return get_tokenizer('meta-llama/Llama-2-13b-chat-hf')
    elif size == 70:
        return get_tokenizer('meta-llama/Llama-2-70b-chat-hf')

def get_response_schema(file: str) -> ResponseSchema:

    with open(file, 'r') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()!='']

    schemas = []
    for i, line in enumerate(lines):
        line = line.split(';')
        if len(line)!=3:
            raise Exception(f'Line {i+1} not properly formatted: {line}')
        schema = ResponseSchema(
            name=line[0].strip(),
            description=line[1].strip(),
            type=line[2].strip()
        )
        schemas.append(schema)

    return schemas


