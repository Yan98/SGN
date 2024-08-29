#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os
os.environ['TRANSFORMERS_CACHE'] = "../../../"
os.environ['HF_HOME'] = "../../../"
import torch
import transformers       
from tqdm import tqdm
from extra import EXTRA

torch.backends.cuda.matmul.allow_tf32 = True

model_name = 'Intel/neural-chat-7b-v3-1'
model =  transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

system_inputs = [
    "You are a biology scientist specialising in gene study. Your mission is to describe the functionality and phenotype of the gene provided by the GeneCards gene symbol from the user. Your descriptions need to be concise and contain keywords only.",
    "You are a biology scientist specialising in gene study. Your mission is to describe the functionality and phenotype of the gene. The descriptions need to be concise and contain keywords only, providing the GeneCards gene symbol and its summary as a helpful reference. Note that the reference most likely contains no information on the functionality and phenotype of the gene.  You are encouraged to complement the missing information for the functionality and phenotype of the gene. Do not directly copy from the reference unless you think it is extremely necessary."
    ]

OUTPUT_DIR = "../name_feature/" + model_name
os.makedirs(OUTPUT_DIR,exist_ok=True)

with torch.no_grad():
    with open("gene2name.pkl", "rb") as f:
        names = pickle.load(f)
        names.update(EXTRA)
        names = {k:v for k,v in names.items() if "symbol" in v and not os.path.exists(os.path.join(OUTPUT_DIR, v["symbol"]+".pkl"))}
        print(len(names))

        for name in tqdm(names):
            LLM_OUTPUT = [[],[]]
            if "symbol" not in names[name]:
                continue
            symbol = names[name]["symbol"]
            if os.path.exists(os.path.join(OUTPUT_DIR, f"{symbol}.pkl")):
                continue
            if "summary" in names[name]:
                summary = names[name]["summary"]
            else:
                summary = None
                
            if summary is not None:
                summary = '.'.join(summary.strip().split('.')[:-2])+ '.'
            PRINT_OUT = True
            for idx in range(2):
                if idx == 0:
                    system_input  = system_inputs[0]
                    user_input = symbol
                else:
                    if summary is None:
                        continue
                    system_input = system_inputs[1]
                    user_input = f"GeneCards gene symbol: {symbol}. Reference: {summary}"
                    
                prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"
                inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).cuda()
                outputs = model.generate(inputs, max_length=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=5)
                
                for output in outputs:
                    response = tokenizer.decode(output, skip_special_tokens=True)
                    response = response.split("### Assistant:\n")[-1].strip()  
                    feature = tokenizer.encode(response, return_tensors="pt", add_special_tokens=False).cuda()
                    feature = model(feature, output_hidden_states=True).hidden_states[-1]
                    LLM_OUTPUT[0].append(response)
                    LLM_OUTPUT[1].append(feature)
                    if PRINT_OUT:
                        print(response, flush=True)
                        PRINT_OUT = False
            with open(os.path.join(OUTPUT_DIR, f"{symbol}.pkl"), "wb") as o:
                pickle.dump(LLM_OUTPUT, o)
