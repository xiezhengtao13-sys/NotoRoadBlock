"""Inference throughput of the proposed network (model only: image pairs and road masks already on the GPU).

Usage (sam3 env): python -I road_v2/bench_inference.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_noto_v2 as v2  # noqa: E402

model = v2.NetV2().to("cuda").eval()
x = torch.rand(64, 3, 224, 224, device="cuda")
road = torch.zeros(64, 1, 224, 224, device="cuda")
road[:, :, 100:124] = 1
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(5):
        model(x, x, road)
    torch.cuda.synchronize()
    t, n = time.time(), 50
    for _ in range(n):
        model(x, x, road)
    torch.cuda.synchronize()
pps = 64 * n / (time.time() - t)
print(f"{pps:.0f} segment pairs per second; {102368 / pps / 60:.1f} min for 102,368 segments "
      f"(batch 64, bf16, model only, {torch.cuda.get_device_name()})")
