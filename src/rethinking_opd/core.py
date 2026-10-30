"""Framework-independent objectives used by the release implementation."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def statewise_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute one full-vocabulary KL value for each prefix state.

    Args:
        student_logits: Student logits with vocabulary on the last dimension.
        teacher_logits: Teacher logits with the same shape as ``student_logits``.
        direction: ``"reverse"`` for KL(student || teacher) or ``"forward"``
            for KL(teacher || student).
        temperature: Shared softmax temperature. The result includes the usual
            temperature-squared correction.
    """
    if direction not in {"reverse", "forward"}:
        raise ValueError(f"unknown KL direction: {direction}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have equal shapes")

    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if direction == "reverse":
        values = student_logp.exp() * (student_logp - teacher_logp)
    else:
        values = teacher_logp.exp() * (teacher_logp - student_logp)
    return values.sum(dim=-1) * (temperature**2)


def first_divergence(
    sampled_codes: torch.Tensor, target_codes: torch.Tensor
) -> torch.Tensor:
    """Return the first mismatched SID decision, or ``L`` for an exact match."""
    if sampled_codes.ndim != 2 or sampled_codes.shape != target_codes.shape:
        raise ValueError("sampled_codes and target_codes must have equal (batch, L) shapes")
    differences = sampled_codes.ne(target_codes)
    length = sampled_codes.shape[1]
    positions = torch.arange(length, device=sampled_codes.device).expand_as(sampled_codes)
    return torch.where(differences, positions, length).min(dim=1).values


def select_target_anchor_rows(
    sampled_codes: torch.Tensor,
    target_codes: torch.Tensor,
    state_budget: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select target-prefix anchors under a matched new-state budget.

    A divergence at level ``d`` adds ``L - d - 1`` previously unseen target
    prefix states. Rows are sampled from lower-cost divergence levels first so
    that the requested state budget is used without diluting the on-policy
    branch. Rows whose only mismatch is the final action add no new state.
    """
    if sampled_codes.ndim != 2 or sampled_codes.shape != target_codes.shape:
        raise ValueError("sampled_codes and target_codes must have equal (batch, L) shapes")
    if sampled_codes.shape[1] < 2:
        raise ValueError("target anchor requires a SID with at least two decisions")
    if state_budget <= 0:
        raise ValueError("state_budget must be positive")

    length = sampled_codes.shape[1]
    forks = first_divergence(sampled_codes, target_codes)
    selected: List[torch.Tensor] = []
    remaining = int(state_budget)

    for fork_level in range(length - 2, -1, -1):
        cost = length - fork_level - 1
        candidates = torch.where(forks == fork_level)[0]
        if candidates.numel() == 0 or remaining < cost:
            continue
        order = torch.randperm(
            candidates.numel(), device=candidates.device, generator=generator
        )
        candidates = candidates.index_select(0, order)
        take = min(candidates.numel(), remaining // cost)
        if take:
            selected.append(candidates[:take])
            remaining -= int(take) * cost

    if not selected:
        empty = torch.empty(0, dtype=torch.long, device=sampled_codes.device)
        return empty, empty
    rows = torch.cat(selected)
    return rows, forks.index_select(0, rows)
