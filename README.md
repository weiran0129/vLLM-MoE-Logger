# vLLM MoE Expert Logger (Qwen1.5-MoE-A2.7B-Chat)
Implement an opt-in, flag-gated Mixture-of-Experts (MoE) logger to vLLM. The logger records a single configurable MoE layer, which experts
are selected for each token along with their router weights (e.g. {layer, token_idx, topk_ids, topk_weights}).

# What we build
* New `MoeLogger` class in `vllm/moe_logger.py`.
* Logging is enabled when `VLLM_LOG_MOE` is set; Default behavior unchanged when logging is off.
* In `vllm/vllm/model_executor/layers/fused_moe/layer.py`, inside `FusedMoELayer.select_experts`, the log hooker insert **right after** the router computes `topk_ids` and `topk_weights`, I call:`self.moe_logger.log_routes(...)`, we log per token.

## About
vLLM: [Installation](https://blog.vllm.ai/2025/01/10/dev-experience.html) | [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart) | [API](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/index.html)
## Getting Started
```bash
# Install
export VLLM_USE_PRECOMPILED=1
pip install -e .
