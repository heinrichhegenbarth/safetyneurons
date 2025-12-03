import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import json
import pandas as pd

# -----------------------Data download-------------------------
# model paths (update if different)
PATH_BASE = "./.ipynb_checkpoints/models/qwen3/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(PATH_BASE, local_files_only=True)

# Load both models
model_base = AutoModelForCausalLM.from_pretrained(
    PATH_BASE, dtype=torch.float16, device_map="auto", local_files_only=True
)


with open("./data/split/test.json") as file:
    train = json.load(file)

df_train = pd.DataFrame(train)
train_prompts = df_train["content"].to_list()

training_labels = df_train["label"].to_list()


# -------------------------------inference--------------------------------

inputs = tokenizer(train_prompts, return_tensors="pt", padding=True).to(
    model_base.device
)

# dictionary for storing activations:
activations_base = {}


def get_hook(activation_dict, name):
    def hook(module, input, output):
        activation_dict[name] = output.detach().float().cpu()

    return hook


# Pick the MLP layers
nlayers = 36
nlayers = min(nlayers, 36)  # ensures we stay within 36 (layers in LLM)
for index in range(nlayers):

    # layer = model.model.layers[LAYER_INDEX]
    layer_base = model_base.model.layers[index]

    # only hooking MLP activations:
    # layer.mlp.register_forward_hook(...)
    hook_base = layer_base.mlp.register_forward_hook(
        get_hook(activations_base, f"layer_{index}")
    )

    # Forward pass - The inference

    with torch.no_grad():
        _ = model_base(
            **inputs
        )  # _ ignoring output since we only care about activations.

    # Remove hooks - important ot reset the hook per iterations (avoiding memory leaks)
    hook_base.remove()


# layers1 (prompts, tokens, neurons)

layers_concat = torch.cat([v for v in activations_base.values()], dim=2)
print(layers_concat.shape)


mean_prompt = layers_concat.mean(dim=1)

print(mean_prompt.shape)


df = pd.DataFrame(mean_prompt)

df.to_csv("test_data.csv")

labels_df = pd.DataFrame(training_labels)

labels_df.to_csv("test_labels.csv")
