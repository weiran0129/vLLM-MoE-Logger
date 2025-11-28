# vLLM MoE Expert Logger (Qwen1.5-MoE-A2.7B-Chat)
Implement an opt-in, flag-gated Mixture-of-Experts (MoE) logger to vLLM. The logger records a single configurable MoE layer, which experts
are selected for each token along with their router weights (e.g. {layer, token_idx, topk_ids, topk_weights}).

## What we build
* New `MoeLogger` class in `vllm/moe_logger.py`.
* Logging is enabled when `VLLM_LOG_MOE` is set; Default behavior unchanged when logging is off.
* In `vllm/vllm/model_executor/layers/fused_moe/layer.py`, inside `FusedMoELayer.select_experts`, the log hooker insert **right after** the router computes `topk_ids` and `topk_weights`, I call:`self.moe_logger.log_routes(...)`, we log per token.

## About
vLLM: A fast and easy-to-use library for LLM inference and serving [Installation](https://blog.vllm.ai/2025/01/10/dev-experience.html) | [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart) | [API](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/index.html)

Model: Qwen/Qwen1.5-MoE-A2.7B-Chat (≈14.3B total, 2.7B activated) [Huggin Face](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)

Prompt Dataset: Use GSM8K test split, first 25 questions [Huggin Face](https://huggingface.co/datasets/openai/gsm8k/tree/main/main)

## Getting Started
System used:

GPU used:

vLLM installation
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
export VLLM_USE_PRECOMPILED=1
pip install -e .
```
Data preparation
```bash
python make_prompts.py
```
Baseline run (no logging, compiled kernels)
```bash
unset VLLM_LOG_MOE
python run_generate.py
