# Reinforcement Learning Finetunes Small Subnetworks in Large Language Models

The purpose of this repository is to provide simple, reproducible tools for analyzing how reinforcement learning fine-tuning affects large language models. Specifically, it measures parameter update sparsity—the fraction of weights that change between a supervised fine-tuned model and its RL-finetuned version—and computes the rank of update matrices to understand the structure and capacity of these changes. This helps researchers study parameter efficiency, interpretability, and compression opportunities in RL-tuned LLMs.

## 🔗 Links

- 📝 [Paper](https://arxiv.org/abs/2505.11711)
- 📰 [Blog (coming)](<your_blog_link_here>)

## Getting Started

1. **Install Requirements**  
   Make sure you have the necessary tools installed:  
   ```bash
   pip install -r requirements.txt
   cd src

2. **Check the update sparsity**
   ```bash
   python check_sparsity.py --sft_model <SFT_MODEL> --rl_model <RL_MODEL> --cache_dir <Your local HF cache> --torch_dtype <bfloat16/float16/..> --device_map cpu --tolerances 1e-5
