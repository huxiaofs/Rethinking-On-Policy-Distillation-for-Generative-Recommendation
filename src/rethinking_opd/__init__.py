"""On-policy distillation utilities for hierarchical generative recommendation."""

from .core import first_divergence, select_target_anchor_rows, statewise_kl

__all__ = ["first_divergence", "select_target_anchor_rows", "statewise_kl"]
