import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

SFT_MODEL = "allenai/Llama-3.1-Tulu-3-8B-SFT"
RL_MODEL  = "allenai/Llama-3.1-Tulu-3-8B-DPO"

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL)

model_sft = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir="/shared/storage-01/huggingface/models/"
)
model_rl = AutoModelForCausalLM.from_pretrained(
    RL_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir="/shared/storage-01/huggingface/models/"
)

all_deltas = []
param_sizes = []
sft_params = []  # references to SFT params so we can update them
rl_state_dict = model_rl.state_dict()
sft_state_dict = model_sft.state_dict()

for name_sft, param_sft in tqdm(model_sft.named_parameters()):
    if name_sft not in rl_state_dict:
        print("woops")
for name_rl, param_rl in tqdm(model_rl.named_parameters()):
    if name_rl not in sft_state_dict:
        print("woops")

delta_dict = {}
with torch.no_grad():
    for name_sft, param_sft in tqdm(model_sft.named_parameters()):
        try:
            param_rl = rl_state_dict[name_sft].to(param_sft.device)
            
            delta =  param_rl - param_sft.data
            delta_dict[name_sft] = delta
        except Exception as e:
            print(e)



from tqdm import tqdm

max_rank = []
rank = []
for key in tqdm(delta_dict.keys()):
    if 'layernorm' in key or len(delta_dict[key].shape)<= 1:
        continue
    rank_ = torch.linalg.matrix_rank(delta_dict[key].float())
    max_rank_ = min(delta_dict[key].shape)
    max_rank.append(max_rank_)
    rank.append(rank_)

    print(rank_, max_rank_)

mean = sum(y / x for x, y in zip(max_rank, rank)) / len(max_rank)
print(mean)