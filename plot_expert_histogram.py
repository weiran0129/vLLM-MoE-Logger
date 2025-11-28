import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt


def load_expert_counts(log_path: str):
    expert_counter = Counter()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "route":
                continue
            for eid in rec["topk_ids"]:
                expert_counter[eid] += 1
    return expert_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to moe_routes JSONL")
    parser.add_argument("--out", default="expert_hist.png", help="Output PNG path")
    args = parser.parse_args()

    counts = load_expert_counts(args.log)
    experts = sorted(counts.keys())
    values = [counts[e] for e in experts]

    plt.figure(figsize=(10, 4))
    plt.bar(experts, values)
    plt.xlabel("Expert ID")
    plt.ylabel("Usage count (token×slot)")
    plt.title(f"Expert usage histogram\n{args.log}")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved histogram to {args.out}")


if __name__ == "__main__":
    main()
