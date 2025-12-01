# TODO model the arcitecture in README.md
# README: only hooking MLP activations:


# label0 = [ex for ex in dataset if ex["label"] == 0]
    # label1 = [ex for ex in dataset if ex["label"] == 1]

    # contrasting_data = label1[:100]  # 100 instances from label 1
    # label0_balanced = label0[100:]  # Drop 100 instances from label 0
    # balanced_data = label1 + label0_balanced

    # train test split
    # generator = torch.Generator().manual_seed(SEED)
    # permutations = torch.randperm(len(balanced_data), generator=generator).tolist()
    # shuffled_data = [balanced_data[i] for i in permutations]
    # split_index = int(0.8 * len(shuffled_data))  # 80 / 20 split

    # train = shuffled_data[:split_index]
    # test = shuffled_data[split_index:]

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# model paths (update if different)
PATH_BASE = "./.ipynb_checkpoints/models/qwen3/Qwen3-4B"
PATH_SAFE = "./.ipynb_checkpoints/models/qwen3/Qwen3-4B-SafeRL"
SEED = 7
tokenizer = AutoTokenizer.from_pretrained(PATH_BASE, local_files_only=True)

# M1
model_base = AutoModelForCausalLM.from_pretrained(
    PATH_BASE, dtype=torch.float16, device_map="auto", local_files_only=True
)
# M2
model_safe = AutoModelForCausalLM.from_pretrained(
    PATH_SAFE, dtype=torch.float16, device_map="auto", local_files_only=True
)

def get_dataset():
    # def tokenize_function(data):
    #     print(data)
    #     print(type(data))
    #     return tokenizer(data["content"], return_tensors="pt", padding=True).to(
    #         model_base.device
    #     )
    train_path = 'data/one_instance/train.json'
    test_path = 'data/one_instance/test.json'
    contrasting_path = 'data/one_instance/contrasting.json'
    # train_path = '.data/split/train.json'
    # test_path = '.data/split/test.json'
    # contrasting_path = '.data/split/contrasting_data.json'
    
    print(train_path, test_path, contrasting_path)

    dataset = load_dataset(
        'json', 
        data_files = {
            'train': train_path, 
            'test': test_path
        }
    )
    
    print(f'This is dataset \n{dataset}')
    contrasting = load_dataset(
        'json', 
        data_files = {
            'contrasting': contrasting_path
        }
        ).remove_columns('label')
    
    print(f'This is the print of contrasting \n{contrasting}')
    # t_contrasting = tokenize_function(contrasting)
    # t_dataset = tokenize_function(dataset)
    
    def preprocessing(ex):
        return tokenizer(
            ex["content"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_attention_mask=True,
        )


    
    print('checkpoint 1')

    t_contrasting = contrasting.map(preprocessing, batched=True)
    print('check 2')
    t_train = dataset['train'].map(preprocessing, batched=True)
    t_test = dataset['test'].map(preprocessing, batched=True)
    
    print('check 3')
    print(t_contrasting)

    print(t_train)
    print(t_test)

    return t_contrasting, t_train, t_test

def activation_contrasting(model, data, nlayers=36):



    def get_hook(activation_dict, name):
        """
        Returns a forward hook that saves module output to activation_dict[name] as a float32 CPU tensor.
        For use with nn.Module.register_forward_hook.
        """

        def hook(module, data, output):
            activation_dict[name] = output.detach().float().cpu()

        return hook


    activations_base = {}
    activations_safe = {}

    # Hooking the MLP layers
    for index in range(nlayers):
        # layer = model.model.layers[LAYER_INDEX]
        layer_base = model_base.model.layers[index]
        layer_safe = model_safe.model.layers[index]

        # layer.mlp.register_forward_hook(...)
        hook_base = layer_base.mlp.register_forward_hook(
            get_hook(activations_base, f"layer_{index}")
        )
        hook_safe = layer_safe.mlp.register_forward_hook(
            get_hook(activations_safe, f"layer_{index}")
        )

        # (Inference) Forward pass
        print(**data)
        with torch.no_grad():
            _ = model_base(**data)  # _ ignoring output since we only care about activations.
            _ = model_safe(**data)

        # Remove hooks - important ot reset the hook per iterations (avoiding memory leaks)
        hook_base.remove()
        hook_safe.remove()
    return activations_base, activations_safe


def compute_change_scores(activations_base, activations_safe, nlayers=1):
    # Computing the RMSE, per neuron per layer:
    rms_layer = []
    for index in range(nlayers):
        diff_sq = (
            activations_base[f"layer_{index}"] - activations_safe[f"layer_{index}"]
        ) ** 2  # [2, 6, 2560]
        rms_layer.append(diff_sq.mean(dim=(0, 1)).sqrt())  # [2560]

    # stacking the layers:
    all_activations = torch.stack(rms_layer)
    print(all_activations.shape)

    # flattening all the activations:
    flat_values = torch.cat(rms_layer)  # tensor of all values
    ids = torch.arange(len(flat_values))  # numeric ids
    result = list(zip(ids.tolist(), flat_values.tolist()))


def topk(list_of_tup, k: float):
    descending = sorted(list_of_tup, key=lambda x: x[int(1)], reverse=True)
    size = int(len(list_of_tup) * k)
    res = descending[:size]

    return res


def main():
    t_contrasting, t_train, t_test = get_dataset()
    print(t_contrasting)
    print(t_train)
    print(t_test)

    activations_base, activations_safe = activation_contrasting(
        model_base, t_contrasting
    )
    result = compute_change_scores(activations_base, activations_safe)
    safety_neurons = topk(result, 0.05)

    print(len(safety_neurons))
    print(safety_neurons)

    # save outputs
    with open("safety_neurons.json", "w") as file:
        json.dump(safety_neurons, file)

    t_contrasting, t_train, t_test = get_dataset()


if __name__ == "__main__":
    main()
