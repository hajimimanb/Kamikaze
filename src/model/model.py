# -*- coding: utf-8 -*-
"""Backward-compatibility shim: canonical implementation lives in model.net.

The multi-head CNN (Suphx-style shared trunk) is implemented in
src/model/net.py (task t4 deliverable). Older imports keep working here.
"""
from model.net import (  # noqa: F401
    BinaryHead,
    DiscardHead,
    MultiHeadRiichiNet,
    ResidualBlock,
    ValueHead,
    estimate_macs,
    get_device,
    model_profile,
)

__all__ = ["BinaryHead", "DiscardHead", "MultiHeadRiichiNet", "ResidualBlock",
           "ValueHead", "estimate_macs", "get_device", "model_profile"]
