from __future__ import annotations

import copy

import hydra
import torch
from omegaconf import DictConfig, open_dict

from src.train import train
from src.utils import extras
from src.utils.custom_hydra_resolvers import *  # noqa: F401,F403
from src.utils.restart_job import LocalJobLauncher

torch.set_float32_matmul_precision("medium")


@hydra.main(
    version_base="1.3",
    config_path="../../third_party/GRID/configs",
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> None:
    teacher_cfg = copy.deepcopy(cfg.model)
    with open_dict(teacher_cfg):
        teacher_cfg._target_ = (
            "src.models.modules.semantic_id.tiger_generation_model."
            "SemanticIDEncoderDecoder"
        )
        teacher_cfg.huggingface_model.config.d_model = 256
        teacher_cfg.huggingface_model.config.d_ff = 2048
        teacher_cfg.huggingface_model.config.num_heads = 8
        teacher_cfg.huggingface_model.config.d_kv = 64
        teacher_cfg.decoder.config.d_model = 256
        teacher_cfg.decoder.config.d_ff = 2048
        teacher_cfg.decoder.config.num_heads = 8
        teacher_cfg.decoder.config.d_kv = 64

    with open_dict(cfg):
        cfg.model._target_ = "rethinking_opd.grid_model.OPDSemanticIDEncoderDecoder"
        cfg.model.teacher = teacher_cfg
        cfg.model.teacher_checkpoint = cfg.opd.teacher_checkpoint
        cfg.model.opd_direction = cfg.opd.direction
        cfg.model.use_target_anchor = cfg.opd.use_target_anchor
        cfg.model.temperature = cfg.opd.temperature
        cfg.model.anchor_state_budget_fraction = cfg.opd.anchor_state_budget_fraction
        cfg.model.sampling_seed = cfg.seed

    extras(cfg)
    LocalJobLauncher(cfg=cfg).launch(function_to_run=train)


if __name__ == "__main__":
    main()
