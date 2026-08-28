# -*- coding: utf-8 -*-
"""Torch CUDA verification for RTX 5070 (Blackwell sm_120).

Requirements: torch cu128+ build (sm_120 kernels only exist in CUDA 12.8+).
Installed via:
  pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130

Checks: version/CUDA runtime, device props, matmul benchmark, and a full
forward+backward pass of MultiHeadRiichiNet (simple mode). Exit 0 = OK.
"""
import os
import sys
import time

import torch

# allow direct execution: python model/check_cuda.py from anywhere
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    print("torch:", torch.__version__, "| cuda runtime:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device. Install cu128+ torch build (see docstring).")
        return 1
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print("device:", props.name, "| cc=" + str(props.major) + "." + str(props.minor),
          "| vram=" + format(props.total_memory / 2 ** 30, ".1f") + " GiB")
    if props.major < 12:
        print("FAIL: pre-Blackwell device, expected sm_120 (RTX 5070).")
        return 1

    for dtype, tag in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
        a = torch.randn(4096, 4096, device=dev, dtype=dtype)
        b = torch.randn(4096, 4096, device=dev, dtype=dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
        print("matmul " + tag + ": " + format(2 * 4096 ** 3 / dt / 1e12, ".2f") + " TFLOPS")

    from model.features import feature_channels
    from model.model import MultiHeadRiichiNet
    c = feature_channels("simple")
    net = MultiHeadRiichiNet(c, channels=256, n_blocks=50).to(dev)
    x = torch.randn(8, c, 34, 1, device=dev)
    mask = torch.ones(8, 34, dtype=torch.bool, device=dev)
    cands = {h: torch.zeros(8, 34, device=dev)
             for h in ("riichi", "chow", "pon", "kan")}
    logits = net(x, masks={"discard": mask}, candidates=cands)
    # exercise every head so all parameters receive gradients
    loss = sum(v.float().sum() for v in logits.values())
    loss.backward()
    torch.cuda.synchronize()
    n_params = net.num_parameters() / 1e6
    ok = all(p.grad is not None for p in net.parameters() if p.requires_grad)
    print("model forward+backward OK: in=" + str(c) + " ch, params="
          + format(n_params, ".2f") + "M, grads_ok=" + str(ok))
    print("CUDA VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
