# %%
# Imports

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import json
import pandas as pd

# %%
# Data download

PATH_BASE = "./models/qwen3/Qwen3-4B"
BATCH_SIZE = 40

tokenizer = AutoTokenizer.from_pretrained(PATH_BASE, local_files_only=True)

# Load base model
model_base = AutoModelForCausalLM.from_pretrained(
    PATH_BASE, dtype=torch.float16, device_map="auto", local_files_only=True
)


# Load training data
with open("./data/split/train.json") as file:
    train = json.load(file)

with open("./data/split/test.json") as file:
    test = json.load(file)

train_prompts = [item["content"] for item in train]
test_prompts = [item["content"] for item in test]
training_labels = [item["label"] for item in train]
test_labels = [item["label"] for item in test]

print(
    f"Training: {len(train_prompts):>6} prompts | {len(training_labels):>6} labels\n"
    f"Test:     {len(test_prompts):>6} prompts | {len(test_labels):>6} labels\n"
)

# RESTRICT DATASET TO 4 PROMPTS FOR TESTING
# train_prompts = train_prompts[:4]
# training_labels = training_labels[:4]
# test_prompts = test_prompts[:4]
# test_labels = test_labels[:4]

# %%
# Inference

def get_inference(prompts, model, tokenizer):
    def get_hook(activation_dict, name):
        def hook(module, input, output):
            activation_dict[name] = output.detach().float().cpu()

        return hook

    # Hooking the MLP layers
    activations = {}
    hooks = []
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    for index in range(36):  # number of layers in the model
        layer = model.model.layers[index]
        hooks.append(
            layer.mlp.register_forward_hook(get_hook(activations, f"layer_{index}"))
        )

    # Forward pass
    # we ignore outputs as we only predict the next token.
    # we are only interested in the activations and not in the model responses
    with torch.no_grad():
        _ = model(**inputs)

    for hook in hooks:
        hook.remove()

    return activations

def build_feature_tensor(activations_all):
    """
    Concatenate batches per layer and then across layers into a single feature tensor.
    Returns a tensor shaped [num_prompts, nlayers*hidden].
    """
    # concatenate batches per layer -> {layer_index: [num_prompts, activations]}
    layer_to_tensor = {
        k: torch.cat(v_list, dim=0) for k, v_list in activations_all.items()
    }
    # order layers by index and concatenate along neuron dimension -> [num_prompts, nlayers*hidden]
    ordered_keys = sorted(layer_to_tensor.keys(), key=lambda s: int(s.split("_")[1]))
    features = torch.cat([layer_to_tensor[k] for k in ordered_keys], dim=1)
    return features


def run_batched_activation_generation(prompts):
    activations_all = {}

    for batch in range(0, len(train_prompts), BATCH_SIZE):
        end = min(batch + BATCH_SIZE, len(train_prompts))
        batch_prompts = train_prompts[batch:end]
        activations = get_inference(batch_prompts, model_base, tokenizer)
        # accumulate per-layer, averaging over tokens to avoid seq-length mismatches
        for k, v in activations.items():  # v: [batch, tokens, hidden]
            activations_all.setdefault(k, []).append(v.mean(dim=1))  # [batch, hidden]
        print(f"Batch {batch}: collected {len(activations)} layers")

    features = build_feature_tensor(activations_all)
    print("Features tensor shape:", features.shape)
    print("Features (dataset):", features.shape[1])

    return features


# %%
# Create and Save Training Data
training_features = run_batched_activation_generation(train_prompts)
df_training_features = pd.DataFrame(training_features.numpy())
df_training_labels = pd.DataFrame(training_labels)
df_training_data = pd.concat([df_training_labels, df_training_features], axis=1)
df_training_data.to_csv("training_data.csv", index=False)

# %%
# Create and Save Testing Data
testing_features = run_batched_activation_generation(test_prompts)
df_testing_features = pd.DataFrame(testing_features.numpy())
df_testing_labels = pd.DataFrame(test_labels)
df_testing_data = pd.concat([df_testing_labels, df_testing_features], axis=1)
df_testing_data.to_csv("testing_data.csv", index=False)