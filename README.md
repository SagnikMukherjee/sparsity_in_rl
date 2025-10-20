# Reinforcement Learning Finetunes Small Subnetworks in Large Language Models
 


## 🔗 Links

- 📝 [Paper](https://arxiv.org/abs/2505.11711)

## Getting Started

1. **Install Requirements**  
   Make sure you have the necessary tools installed:  
   ```bash
   pip install -r requirements.txt
   cd src

2. **Check the update sparsity**
   ```bash
   python check_sparsity.py --sft_model <SFT_MODEL> --rl_model <RL_MODEL> --cache_dir <Your local HF cache> --torch_dtype <bfloat16/float16/..> --device_map cpu --tolerances 1e-5
