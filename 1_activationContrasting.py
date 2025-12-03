import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import json
import pandas as pd

# ------------------------------- Data Download -------------------------------#

# model paths (update if different)
PATH_BASE = "./.ipynb_checkpoints/models/qwen3/Qwen3-4B"
PATH_SAFE = "./.ipynb_checkpoints/models/qwen3/Qwen3-4B-SafeRL"

tokenizer = AutoTokenizer.from_pretrained(PATH_BASE, local_files_only=True)

# Load both models
model_base = AutoModelForCausalLM.from_pretrained(
    PATH_BASE, dtype=torch.float16, device_map="auto", local_files_only=True
)

model_safe = AutoModelForCausalLM.from_pretrained(
    PATH_SAFE, dtype=torch.float16, device_map="auto", local_files_only=True
)

with open("/data/split/contrasting_data.json") as file:
    harm100 = json.load(file)

df_prompt = pd.DateFrame(harm100)
harm_prompts = df_prompt["content"].to_list()


# ------------------------------- inference ------------------------------#

inputs = tokenizer(harm_prompts, return_tensors="pt", padding=True).to(
    model_base.device
)

# dictionary for storing activations:
activations_base = {}
activations_safe = {}


def get_hook(activation_dict, name):
    def hook(module, input, output):
        activation_dict[name] = output.detach().float().cpu()  # the activations

    return hook


# Pick the MLP layers
nlayers = 36
nlayers = min(nlayers, 36)  # ensures we stay within 36 (layers in LLM)
for index in range(nlayers):

    # layer = model.model.layers[LAYER_INDEX]
    layer_base = model_base.model.layers[index]
    layer_safe = model_safe.model.layers[index]

    # only hooking MLP activations:
    # layer.mlp.register_forward_hook(...)
    hook_base = layer_base.mlp.register_forward_hook(
        get_hook(activations_base, f"layer_{index}")
    )
    hook_saferl = layer_safe.mlp.register_forward_hook(
        get_hook(activations_safe, f"layer_{index}")
    )

    # Forward pass - The inference

    with torch.no_grad():
        _ = model_base(
            **inputs
        )  # _ ignoring output since we only care about activations.
        _ = model_safe(**inputs)

    # Remove hooks - important ot reset the hook per iterations (avoiding memory leaks)
    hook_base.remove()
    hook_saferl.remove()


# ------------------------------- Activation contrasting -------------------------------#

rms_layer = []
for i in range(nlayers):
    diff_sq = (
        activations_base[f"layer_{i}"] - activations_safe[f"layer_{i}"]
    ) ** 2  # [2, 6, 2560]
    rms_layer.append(diff_sq.mean(dim=(0, 1)).sqrt())

# show top rms values per layer
for i in range(len(rms_layer)):
    print(rms_layer[i].sort()[0][-1])

# stacking the layers:
all_activations = torch.stack(rms_layer)
all_activations.shape

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(all_activations.numpy(), annot=False, cmap="viridis")
plt.show()


# flattening all the activations:

flat_values = torch.cat(rms_layer)  # tensor of all values
ids = torch.arange(len(flat_values))  # numeric ids
result = list(zip(ids.tolist(), flat_values.tolist()))


# ------------------------------- Selecting top activations -------------------------------#


def topk(list_of_tup, k: float):
    """
    function to return top k of safety neurons

    Args:
        tup: tuple containing in the form (index, value)
        k: the top k% of safety neurons we want to get

    returns:
        res: the resulting top k% safety neurons
    """
    descending = sorted(list_of_tup, key=lambda x: x[int(1)], reverse=True)
    size = int(len(list_of_tup) * k)
    res = descending[:size]

    return res


top5 = topk(result, 0.05)
print(len(result))
print(top5)
