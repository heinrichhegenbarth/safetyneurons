import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import json
import pandas as pd

#-----------------------Data download-------------------------
# model paths (update if different)
PATH_BASE = "./.ipynb_checkpoints/models/qwen3/Qwen3-4B"                   

tokenizer = AutoTokenizer.from_pretrained(PATH_BASE, local_files_only=True)

# Load both models
model_base = AutoModelForCausalLM.from_pretrained(PATH_BASE, 
            dtype=torch.float16, 
            device_map="auto",
            local_files_only=True)




with open('dataset.json', 'r') as file:
    input = json.load(file)


#saving as df
df_prompt = pd.DataFrame(input)


#subsetting 100 harmfull prompts and removing from df 

harmfull100 = df_prompt[df_prompt['label']==1].sample(n=100)
df_prompt = df_prompt.drop(harmfull100.index).reset_index(drop=True)

#list of prompts for inference: 
harm_promps = harmfull100['content'].to_list()


#rebalancing
balance = df_prompt[df_prompt['label']==0].sample(n=100)
df_prompt = df_prompt.drop(balance.index).reset_index(drop=True)

print(f'final df shape: {df_prompt.shape}')
print(f'shape of harmfull subset: {harmfull100.shape}')

#shufling df
df_prompt.sample(frac=1).reset_index(drop=True)

prompts = df_prompt['content']