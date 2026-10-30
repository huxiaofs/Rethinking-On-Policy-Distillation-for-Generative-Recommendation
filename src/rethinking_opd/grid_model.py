from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import src.utils  # Initialize GRID utilities before its circular interface imports.

from .core import select_target_anchor_rows, statewise_kl

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)


class OPDSemanticIDEncoderDecoder(SemanticIDEncoderDecoder):
    """GRID student trained on student-rollout states with a frozen teacher.

    The teacher is deliberately kept outside the registered module tree so it
    is excluded from the optimizer and student checkpoints.  This preserves
    exact compatibility with the student-5k Lightning checkpoint, including
    its Adam state and global step.
    """

    def __init__(
        self,
        teacher: SemanticIDEncoderDecoder,
        teacher_checkpoint: str,
        opd_direction: str,
        use_target_anchor: bool = False,
        temperature: float = 1.0,
        anchor_state_budget_fraction: float = 1.0,
        sampling_seed: int = 42,
        **kwargs,
    ) -> None:
        if opd_direction not in {"reverse", "forward"}:
            raise ValueError("opd_direction must be reverse or forward")
        if not 0.0 < anchor_state_budget_fraction <= 1.0:
            raise ValueError("anchor_state_budget_fraction must be in (0, 1]")
        super().__init__(**kwargs)

        # The GRID base class introspects the outer constructor frame when it
        # saves hyperparameters.  Remove the concrete teacher object again: it
        # must not be pickled into student checkpoints.
        for container_name in ("_hparams", "_hparams_initial"):
            container = getattr(self, container_name, None)
            if container is not None:
                container.pop("teacher", None)

        checkpoint = torch.load(
            teacher_checkpoint, map_location="cpu", weights_only=False
        )
        teacher.load_state_dict(checkpoint["state_dict"], strict=True)
        teacher.requires_grad_(False)
        teacher.eval()
        # Bypass nn.Module.__setattr__: the frozen teacher must not enter the
        # student state_dict or optimizer parameter list.
        object.__setattr__(self, "_opd_teacher", teacher)
        self.opd_direction = opd_direction
        self.use_target_anchor = bool(use_target_anchor)
        self.temperature = float(temperature)
        self.anchor_state_budget_fraction = float(anchor_state_budget_fraction)
        self.sampling_seed = int(sampling_seed)

    @property
    def opd_teacher(self) -> SemanticIDEncoderDecoder:
        return object.__getattribute__(self, "_opd_teacher")

    def on_fit_start(self) -> None:
        self.opd_teacher.to(self.device)
        self.opd_teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The hidden teacher is not reached by Module.train(), but keeping the
        # invariant explicit makes accidental future registration harmless.
        if "_opd_teacher" in self.__dict__:
            self.opd_teacher.eval()
        return self

    @staticmethod
    def _model_inputs(
        model_input: SequentialModelInputData,
        feature_to_model_input_map: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        return {
            feature_to_model_input_map.get(key, key): value
            for key, value in model_input.transformed_sequences.items()
        }

    @staticmethod
    def _target_codes(
        model_input: SequentialModelInputData,
        label_data: SequentialModuleLabelData,
    ) -> torch.Tensor:
        target = next(iter(label_data.labels.values()))
        return target.reshape(model_input.mask.size(0), -1).long()

    @staticmethod
    def _next_logits(
        model: SemanticIDEncoderDecoder,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        prefix: Optional[torch.Tensor],
        level: int,
    ) -> torch.Tensor:
        decoder_output = model.decoder_forward_pass(
            future_ids=prefix,
            encoder_output=encoder_output,
            attention_mask_for_encoder=encoder_mask,
            use_cache=False,
        )
        return model.decoder.decoder_mlp[level](decoder_output[:, -1])

    def _encode(
        self,
        model: SemanticIDEncoderDecoder,
        model_input: SequentialModelInputData,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mapped = self._model_inputs(model_input, model.feature_to_model_input_map)
        return model.encoder_forward_pass(
            attention_mask=model_input.mask,
            input_ids=mapped["input_ids"],
            user_id=mapped.get("user_id"),
        )

    def _sample_and_score_main(
        self,
        model_input: SequentialModelInputData,
        generator: torch.Generator,
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        student_memory, student_mask = self._encode(self, model_input)
        with torch.no_grad():
            teacher_memory, teacher_mask = self._encode(self.opd_teacher, model_input)

        sampled_actions: List[torch.Tensor] = []
        student_logits: List[torch.Tensor] = []
        teacher_logits: List[torch.Tensor] = []
        for level in range(self.num_hierarchies):
            prefix = (
                None if level == 0 else torch.stack(sampled_actions, dim=1).detach()
            )
            student_level = self._next_logits(
                self, student_memory, student_mask, prefix, level
            )
            with torch.no_grad():
                teacher_level = self._next_logits(
                    self.opd_teacher,
                    teacher_memory,
                    teacher_mask,
                    prefix,
                    level,
                )
            probabilities = F.softmax(
                student_level.detach().float() / self.temperature, dim=-1
            )
            action = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            ).squeeze(1)
            sampled_actions.append(action)
            student_logits.append(student_level)
            teacher_logits.append(teacher_level)

        return (
            torch.stack(sampled_actions, dim=1),
            student_logits,
            teacher_logits,
            (student_memory, student_mask),
            (teacher_memory, teacher_mask),
        )

    def _anchor_losses(
        self,
        target_codes: torch.Tensor,
        selected_rows: torch.Tensor,
        fork_levels: torch.Tensor,
        student_encoded: Tuple[torch.Tensor, torch.Tensor],
        teacher_encoded: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        losses: List[torch.Tensor] = []
        level_losses: List[torch.Tensor] = []
        student_memory, student_mask = student_encoded
        teacher_memory, teacher_mask = teacher_encoded
        for level in range(1, self.num_hierarchies):
            valid = fork_levels < level
            rows = selected_rows[valid]
            if rows.numel() == 0:
                level_losses.append(torch.zeros((), device=target_codes.device))
                continue
            prefix = target_codes.index_select(0, rows)[:, :level]
            student_level = self._next_logits(
                self,
                student_memory.index_select(0, rows),
                student_mask.index_select(0, rows),
                prefix,
                level,
            )
            with torch.no_grad():
                teacher_level = self._next_logits(
                    self.opd_teacher,
                    teacher_memory.index_select(0, rows),
                    teacher_mask.index_select(0, rows),
                    prefix,
                    level,
                )
            state_losses = statewise_kl(
                student_level, teacher_level, "forward", self.temperature
            )
            losses.append(state_losses)
            level_losses.append(state_losses.mean())
        return losses, level_losses

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        model_input, label_data = batch[0]
        local_rank = int(getattr(self, "local_rank", 0))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            self.sampling_seed
            + local_rank * 1_000_003
            + int(self.global_step) * 10_007
        )

        (
            sampled_codes,
            student_logits,
            teacher_logits,
            student_encoded,
            teacher_encoded,
        ) = self._sample_and_score_main(model_input, generator)

        main_state_losses = [
            statewise_kl(student, teacher, self.opd_direction, self.temperature)
            for student, teacher in zip(student_logits, teacher_logits)
        ]
        main_loss = torch.cat(main_state_losses).mean()
        anchor_sum = torch.zeros((), device=self.device)
        anchor_states = torch.zeros((), device=self.device)
        anchor_level_losses = [torch.zeros((), device=self.device) for _ in range(3)]

        if self.use_target_anchor:
            targets = self._target_codes(model_input, label_data)
            budget = max(
                1,
                int(round(targets.shape[0] * self.anchor_state_budget_fraction)),
            )
            selected_rows, fork_levels = select_target_anchor_rows(
                sampled_codes, targets, budget, generator=generator
            )
            if selected_rows.numel():
                losses, by_level = self._anchor_losses(
                    targets,
                    selected_rows,
                    fork_levels,
                    student_encoded,
                    teacher_encoded,
                )
                if losses:
                    anchor_sum = torch.cat(losses).sum()
                    anchor_states = torch.tensor(
                        float(sum(loss.numel() for loss in losses)), device=self.device
                    )
                for offset, value in enumerate(by_level, start=1):
                    anchor_level_losses[offset] = value

        # Main-preserving additive objective: no on-policy state is diluted.
        main_states = float(sum(loss.numel() for loss in main_state_losses))
        loss = main_loss + anchor_sum / main_states

        self.train_loss(loss)
        log_values = {
            "train/loss": self.train_loss,
            "train/main_kl": main_loss.detach(),
            "train/anchor_kl_additive": (anchor_sum / main_states).detach(),
            "train/anchor_states": anchor_states.detach(),
        }
        for level, level_loss in enumerate(main_state_losses):
            log_values[f"train/main_kl_l{level}"] = level_loss.mean().detach()
            log_values[f"train/anchor_kl_l{level}"] = anchor_level_losses[level].detach()
        self.log_dict(
            log_values,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss
