from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer


# loading model locally
tokenizer = AutoTokenizer.from_pretrained("../models/qwen3/Qwen3-4B", local_files_only=True)
processor = AutoProcessor.from_pretrained("../models/qwen3/Qwen3-4B", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("../models/qwen3/Qwen3-4B", local_files_only=True)

# running inference on the model
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))