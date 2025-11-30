# user and assistant on data

import json
import os

import torch
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoProcessor,
)

# ------------------------------------------------------------

# API Keys
WANDB_API_KEY = "9a14945f8b0ff66fea5fa2806b008bb4073feef4"

# Paths
MODEL_PATH = "./models/qwen3/Qwen3-4B"
DATA_PATH = "./data/formatted_instruction_data.json"
OUTPUT_DIR = "./models/finetuned/qwen3-4b-sft"

# Settings
MAX_SEQ_LENGTH = 2048
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 2
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


# ------------------------------------------------------------

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            samples.append(json.loads(line))
    return samples

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_seq_length,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        messages = self.data[index]

        # Tokenize user-only (prompt) to find boundary
        prompt_messages = messages[:-1]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )[0]

        # Tokenize full conversation (user + assistant)
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
        )[0]

        # Right-truncate both prompt and full sequences for consistent boundaries
        if prompt_ids.shape[0] > self.max_seq_length:
            prompt_ids = prompt_ids[: self.max_seq_length]
        if full_ids.shape[0] > self.max_seq_length:
            full_ids = full_ids[: self.max_seq_length]

        # Mask prompt tokens
        prompt_len = min(prompt_ids.shape[0], full_ids.shape[0])

        labels = full_ids.clone()
        labels[:prompt_len] = -100

        attention_mask = torch.ones_like(full_ids)

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CustomDataCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        input_ids_list = [feature["input_ids"].tolist() for feature in features]
        attention_masks_list = [feature["attention_mask"].tolist() for feature in features]
        labels_list = [feature["labels"] for feature in features]

        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attention_masks_list},
            padding=True,
            return_tensors="pt",
        )
        max_len = padded["input_ids"].shape[1]
        padded_labels = torch.full(
            (len(labels_list), max_len), self.label_pad_token_id, dtype=torch.long
        )
        for index, label in enumerate(labels_list):
            seq_len = min(label.shape[0], max_len)
            padded_labels[index, :seq_len] = label[:seq_len]

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": padded_labels,
        }


def main():
    random_seed = 7
    torch.manual_seed(random_seed)

    # setup cuda
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    print(f"tokenizer pad token: {tokenizer.pad_token}")
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

    # double check the target modules
    # doube check the config with chen paper
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    # Load dataset

    # assumption that the data is already in the format {user: ..., assistant: ...}
    data = load_jsonl(DATA_PATH)

    print(f"Loaded {len(data)} samples from {DATA_PATH}")
    print(f"Data Instance: {data[0]}")

    train_dataset = SFTDataset(data, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    # wandb
    report_to = []
    if WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        report_to = ["wandb"]

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()

