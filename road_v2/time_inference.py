"""How long does scoring the whole road network take? Inference throughput of a ConvNeXt-T classifier.

Measures images per second for 224 x 224 inputs (batch 64, bf16) on the local GPU, the post-only model and the
Siamese model (two images per segment), and extrapolates to the 102,368 segments of the study area. Image download
and cropping are not included.

Usage:
  python -I road_v2/time_inference.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time
from pathlib import Path

import timm
import torch

sys.stdout.reconfigure(encoding="utf-8")
N_NETWORK = 102368


def main() -> int:
    dev = torch.device("cuda")
    model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, num_classes=1).to(dev).eval()
    x = torch.randn(64, 3, 224, 224, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5):
            model(x)
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        while time.time() - t0 < 20:
            model(x); n += 64
        torch.cuda.synchronize(); ips = n / (time.time() - t0)
    out = dict(gpu=torch.cuda.get_device_name(0), images_per_s=round(ips), post_minutes=round(N_NETWORK / ips / 60, 1),
               siamese_minutes=round(2 * N_NETWORK / ips / 60, 1))
    print(json.dumps(out))
    Path(__file__).resolve().parent.joinpath("results_noto", "inference_time.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
