"""Standalone validation entry point for the pinned GRID adapter."""

from typing import Any, Dict, Optional, Tuple

import hydra
from omegaconf import DictConfig

from src.utils import RankedLogger, extras
from src.utils.custom_hydra_resolvers import *  # noqa: F401,F403
from src.utils.launcher_utils import pipeline_launcher
from src.utils.restart_job import LocalJobLauncher

log = RankedLogger(__name__, rank_zero_only=True)


def validate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with pipeline_launcher(cfg) as modules:
        log.info("Starting standalone validation")
        modules.trainer.validate(
            model=modules.model,
            datamodule=modules.datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )
        metrics = dict(modules.trainer.callback_metrics)
        log.info("Validation metrics: %s", metrics)
        return metrics, {}


@hydra.main(
    version_base="1.3",
    config_path="../../third_party/GRID/configs",
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    LocalJobLauncher(cfg=cfg).launch(function_to_run=validate)
    return None


if __name__ == "__main__":
    main()
