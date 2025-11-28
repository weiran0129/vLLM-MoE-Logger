# inspect_moe_layers.py
from transformers import AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,        # use dtype instead of torch_dtype, CPU is fine
    low_cpu_mem_usage=False,    # avoid fancy loading that might need accelerate
)

moe_layers = []
layers = model.model.layers  # Qwen2Moe architecture

for tf_idx, block in enumerate(layers):
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        continue

    # Heuristic: MoE layers usually expose experts / num_experts
    if hasattr(mlp, "experts") or hasattr(mlp, "num_experts"):
        moe_layers.append(tf_idx)

print("Transformer layers that are MoE:", moe_layers)
print()
for logical_idx, tf_idx in enumerate(moe_layers):
    print(f"logical MoE layer {logical_idx} -> transformer layer {tf_idx}")
