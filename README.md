# vLLM MoE Expert Logger (Qwen1.5-MoE-A2.7B-Chat)
Implement an opt-in, flag-gated Mixture-of-Experts (MoE) logger to vLLM. The logger records a single configurable MoE layer, which experts
are selected for each token along with their router weights (e.g. {layer, token_idx, topk_ids, topk_weights}).
