import json
import os
import random
import time

from vllm import LLM, SamplingParams

# Fixed seed for reproducibility
SEED = 1234
random.seed(SEED)

prompts = open("prompts.txt", encoding="utf-8").read().split("\n\n---\n\n")

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

# Optional: let logger know which model/seed to record in the meta header
os.environ.setdefault("VLLM_MOE_MODEL_ID", MODEL_ID)
os.environ.setdefault("VLLM_MOE_SEED", str(SEED))

# ---- NEW: toggle eager mode when logging is enabled ----
LOGGING_ENABLED = "VLLM_LOG_MOE" in os.environ

llm = LLM(
    model=MODEL_ID,
    max_model_len=512,  # keep it small
    enforce_eager=LOGGING_ENABLED,  # avoid CUDA graph capture when logging
    disable_log_stats=True,         # optional, keeps vLLM logs cleaner
)

def run_and_time(key: str, timing: dict):
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params)
    t1 = time.time()

    tokens_generated = sum(len(o.outputs[0].token_ids) for o in outs)
    timing[key] = {
        "wall_time_sec": t1 - t0,
        "tokens_generated": tokens_generated,
    }


def main():
    timing_path = "timing.json"
    timing = {}

    if os.path.exists(timing_path):
        timing = json.load(open(timing_path, encoding="utf-8"))

    # When VLLM_LOG_MOE is not set, this run is "no_log".
    # When it IS set, this run is "log".
    if "VLLM_LOG_MOE" in os.environ:
        run_and_time("log", timing)
    else:
        run_and_time("no_log", timing)

    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)


if __name__ == "__main__":
    main()