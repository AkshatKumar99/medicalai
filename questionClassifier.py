import sys

import os
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from functools import partial

# Langchain Imports
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain import PromptTemplate, HuggingFacePipeline

from langchain_core.exceptions import OutputParserException

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from custom_classes import BooleanJsonParser, PromptDataset

import csv

def options_to_txt(answers):

    ans_str = ''
    for choice, txt in answers.items():
        ans_str += choice+'. '+txt+'\n'

    return ans_str

def append_dict_to_csv(dictionary, csv_filename):
    # Check if the file already exists
    file_exists = True
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            if not any(reader):
                file_exists = False
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in append mode
    with open(csv_filename, 'a', newline='') as file:
        # Define the fieldnames including 'count'
        fieldnames = list(dictionary.keys()) + ['count']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writeheader()

        # Get the current count by reading the number of existing rows
        with open(csv_filename, 'r') as file:
            count = len(list(csv.reader(file))) if file_exists else 0

        # Increment the count for the new row
        count += 1
        dictionary['count'] = count

        # Write the values
        writer.writerow(dictionary)

def main():

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", help="Which GPUs to use. Enter using ',' as a separator", required=True, type=str)
    parser.add_argument('-lls', '--llama_size', help="The size of the llama model to use. Can use the 7B, 13B, or 70B parameter model", choices=[7, 13, 70], required=True, type=int)
    parser.add_argument('-bs', '--batch_size', help='The batch size to put into the model', type=int, const=1, nargs='?', default=5)
    parser.add_argument('-i', '--data', help="The path to the MedQA US Qbank", required=True, type=str)
    parser.add_argument('-o', '--output', help="The path to the output file", required=True, type=str)
    parser.add_argument('--step1', help='Runs the script only on Step 1 Questions', action='store_true')
    parser.add_argument('--step2', help='Runs the script only on Step 2 Questions', action='store_true')
    args = parser.parse_args()

    # Set Available GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    from functions import (
        get_response_schema, 
        get_llama_prompt, 
        set_visible_gpus, 
        get_llama_tokenizer, 
        get_llama_pipe,
        get_text_file
    )
    set_visible_gpus(args.gpus)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline
    print(f'\nUsing {torch.cuda.device_count()} GPU(s).')

    print(f'Importing {args.llama_size}B Model...')
    tokenizer = get_llama_tokenizer(args.llama_size)
    pipe = get_llama_pipe(args.llama_size, do_sample=False, num_return_sequences=1, top_k=2, max_new_tokens=500)
    print(f'Imported {args.llama_size}B size model\n')

    data = pd.read_json(args.data, lines=True)
    if args.step1:
        data = data.query("meta_info=='step1'").copy()
        print('Using only Step 1 Questions')
    elif args.step2:
        data = data.query("meta_info=='step2'").copy()
        print('Using only Step 2 Questions')
    else:
        print('Using all both Step 1 and Step 2 Questions')

    SYS_TEMP = get_text_file('templates/sys_template.txt')
    TEMP = get_text_file('templates/temp.txt')

    schema = get_response_schema('templates/classifier_schema.txt')
    output_parser = StructuredOutputParser.from_response_schemas(schema)
    prompt = get_llama_prompt(temp=TEMP, sys_prompt=SYS_TEMP, input_vars=['question', 'answers', 'format_instructions'])
    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())

    prompts = PromptDataset(data = data, answer_transforms=[options_to_txt])
    prompt_loader = DataLoader(prompts, batch_size=args.batch_size, shuffle=False)

    llm = HuggingFacePipeline(pipeline=pipe)
    chain = prompt | llm | BooleanJsonParser()

    print(f'Using a Batch Size of {args.batch_size}')
    classifications = []
    for questions, answers, corrects in tqdm(prompt_loader):
        current_input = [{'question': ques, 'answers': ans} for ques, ans in zip(questions, answers)]
        current_output = chain.batch(current_input)
        for i, out in enumerate(current_output):
            out['question'] = questions[i]
            out['answers'] = answers[i]
            out['correct_answer'] = corrects[i]
            append_dict_to_csv(out, args.output)

if __name__ == '__main__':
    main()






