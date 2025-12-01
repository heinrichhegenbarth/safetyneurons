# TODO model the arcitecture in README.md
# README: only hooking MLP activations:

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model paths (update if different)
PATH_BASE = "./.ipynb_checkpoints/models/qwen3/Qwen3-4B"
PATH_SAFE = "./.ipynb_checkpoints/models/qwen3/Qwen3-4B-SafeRL"
DATA_PATH = "./data/safe&unsafe/dataset.json"
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


def get_dataset(json_path=DATA_PATH):
    def tokenize_function(data_list):
        assert type(data_list) == list, "data_item must be a list"
        return tokenizer(data_list["content"], return_tensors="pt", padding=True).to(
            model_base.device
        )

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    label0 = [ex for ex in dataset if ex["label"] == 0]
    label1 = [ex for ex in dataset if ex["label"] == 1]

    contrasting_data = label1[:100]  # 100 instances from label 1
    label0_balanced = label0[100:]  # Drop 100 instances from label 0
    balanced_data = label1 + label0_balanced

    # train test split
    generator = torch.Generator().manual_seed(SEED)
    permutations = torch.randperm(len(balanced_data), generator=generator).tolist()
    shuffled_data = [balanced_data[i] for i in permutations]
    split_index = int(0.8 * len(shuffled_data))  # 80 / 20 split

    train = shuffled_data[:split_index]
    test = shuffled_data[split_index:]

    with open("data/split/train.json", "w") as file:
        json.dump(train, file)
    with open("data/split/test.json", "w") as file:
        json.dump(test, file)
    with open("data/split/contrasting_data.json", "w") as file:
        json.dump(contrasting_data, file)

    t_train = tokenize_function(train)
    t_test = tokenize_function(data_item)
    t_contrasting = tokenize_function(contrasting_data)

    print(
        f"successfully tokenized {len(t_train)} train instances, {len(t_test)} test instances, {len(t_contrasting)} contrasting instances"
    )
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
        with torch.no_grad():
            _ = model_base(
                **data
            )  # _ ignoring output since we only care about activations.
            _ = model_safe(**data)

        # Remove hooks - important ot reset the hook per iterations (avoiding memory leaks)
        hook_base.remove()
        hook_safe.remove()
    return activations_base, activations_safe


def compute_change_scores(activations_base, activations_safe, nlayers=36):
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
    # t_contrasting, t_train, t_test = get_dataset()
    # print(t_contrasting)
    # print(t_train)
    # print(t_test)

    # activations_base, activations_safe = activation_contrasting(
    #     model_base, t_contrasting
    # )
    # result = compute_change_scores(activations_base, activations_safe)
    # safety_neurons = topk(result, 0.05)

    # print(len(safety_neurons))
    # print(safety_neurons)

    # # save outputs
    # with open("safety_neurons.json", "w") as file:
    #     json.dump(safety_neurons, file)

    t_contrasting, t_train, t_test = get_dataset()


if __name__ == "__main__":
    main()
