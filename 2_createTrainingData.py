import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import json
import pandas as pd

#%%
# Data download

PATH_BASE = "./models/qwen3/Qwen3-4B"
BATCH_SIZE = 100

tokenizer = AutoTokenizer.from_pretrained(PATH_BASE, local_files_only=True)

# Load base model
model_base = AutoModelForCausalLM.from_pretrained(
    PATH_BASE, dtype=torch.float16, device_map="auto", local_files_only=True
)


# Load training data
with open("./data/split/train.json") as file:
    train = json.load(file)

df_train = pd.DataFrame(train)
train_prompts = df_train["content"].to_list()

training_labels = df_train["label"].to_list()


#%%
# Inference


# dictionary for storing activations:
activations_base = {}


def get_hook(activation_dict, name):
    def hook(module, input, output):
        activation_dict[name] = output.detach().float().cpu()

    return hook


# Hooking the MLP layers
nlayers = 36
nlayers = min(nlayers, 36)  # ensures we stay within 36 (layers in LLM)
for index in range(nlayers):
    layer_base = model_base.model.layers[index]
    layer_outputs = []
    for start in range(0, len(train_prompts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(train_prompts))
        batch_prompts = train_prompts[start:end]
        inputs_batch = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
            model_base.device
        )
        batch_store = {}
        hook = layer_base.mlp.register_forward_hook(get_hook(batch_store, "out"))
        with torch.no_grad():
            _ = model_base(**inputs_batch)
        hook.remove()
        layer_outputs.append(batch_store["out"])
    activations_base[f"layer_{index}"] = torch.cat(layer_outputs, dim=0)


# Flatten the layers to have a table with all neuron activations for all prompts
# layers1 (prompts, tokens, neurons)

layers_concat = torch.cat([v for v in activations_base.values()], dim=2)
print(layers_concat.shape)


mean_prompt = layers_concat.mean(dim=1)

# In 2_createTestingData.py, after mean_prompt:
print("Features (dataset):", mean_prompt.shape[1])

#%%
# Save the dataset

df = pd.DataFrame(mean_prompt)

df.to_csv("training_data.csv")

labels_df = pd.DataFrame(training_labels)

labels_df.to_csv("training_labels.csv")
