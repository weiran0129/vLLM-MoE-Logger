# vllm/moe_logger.py
import json
import os
import threading
import time
from typing import Optional

import torch
import vllm

_LOGGER_SINGLETON = None
_LOGGER_LOCK = threading.Lock()


def _get_layer_to_log() -> int:
    """Which logical MoE layer index to log (0-based)."""
    v = os.getenv("VLLM_MOE_LAYER")
    if v is None:
        return 0  # default: first MoE layer
    try:
        return int(v)
    except ValueError:
        return 0


class MoeLogger:
    """Simple JSONL logger for MoE routing decisions."""

    def __init__(self, path: str, layers_logged, top_k: int):
        self.path = path
        self.layers_logged = list(layers_logged)
        self.top_k = int(top_k)

        # Open line-buffered so each write hits disk quickly
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(self.path, "a", buffering=1, encoding="utf-8")

        self._write_meta_once()

    @staticmethod
    def is_enabled() -> bool:
        """Logger is enabled iff VLLM_LOG_MOE is set."""
        return os.getenv("VLLM_LOG_MOE") is not None

    def _write_meta_once(self) -> None:
        """Write the required meta header line."""
        model_id = os.getenv("VLLM_MOE_MODEL_ID", "")
        seed_env = os.getenv("VLLM_MOE_SEED", "-1")
        try:
            seed = int(seed_env)
        except ValueError:
            seed = -1

        meta = {
            "type": "meta",
            "model_id": model_id,
            "vllm_version": getattr(vllm, "__version__", ""),
            "torch_version": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": seed,
            "layers_logged": self.layers_logged,
            "top_k": self.top_k,
            "created_at": time.time(),
        }
        self._file.write(json.dumps(meta) + "\n")

    def log_routes(
        self,
        layer: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        req_id: Optional[str] = None,
    ) -> None:
        """Log one JSONL record per token for this layer."""
        if req_id is None:
            req_id = "r0"

        # Make sure tensors are on CPU and vanilla Python types
        ids = topk_ids.detach().cpu().tolist()
        weights = topk_weights.detach().cpu().tolist()

        for token_idx, (ids_row, w_row) in enumerate(zip(ids, weights)):
            rec = {
                "type": "route",
                "req_id": req_id,
                "token_idx": token_idx,
                "layer": int(layer),
                "topk_ids": ids_row,
                "topk_weights": w_row,
            }
            self._file.write(json.dumps(rec) + "\n")


def get_moe_logger(top_k: int):
    """
    Return the singleton logger (or None if disabled).

    The logged layer index comes from the env var VLLM_MOE_LAYER.
    """
    global _LOGGER_SINGLETON

    if not MoeLogger.is_enabled():
        return None

    layer_to_log = _get_layer_to_log()

    with _LOGGER_LOCK:
        if _LOGGER_SINGLETON is None:
            path = os.environ["VLLM_LOG_MOE"]
            _LOGGER_SINGLETON = MoeLogger(
                path=path,
                layers_logged=[layer_to_log],
                top_k=top_k,
            )
    return _LOGGER_SINGLETON

