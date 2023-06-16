from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

import torch


device = torch.device("cuda")

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

encoding = tokenizer("Tell me what the weather is like today.", truncation=True, return_tensors="pt")
encoding.pop("token_type_ids")
encoding["input_ids"] = torch.hstack((encoding["input_ids"], torch.tensor(tokenizer.eos_token_id).view(1, -1)))
encoding["attention_mask"] = torch.hstack((encoding["attention_mask"], torch.tensor(1).view(1, -1)))
print(encoding)

generate_kwargs = {'max_length': 200, 'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': 11}

output = model.generate(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"], **generate_kwargs)

tokenizer.decode(output[0])