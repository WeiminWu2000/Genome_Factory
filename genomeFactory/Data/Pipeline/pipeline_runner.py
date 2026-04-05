"""
Multi-stage bioinformatics pipeline runner.

Users define pipeline stages as PipelineStage subclasses with a
``process(input_dir, output_dir, config)`` method, then chain them in a
YAML config.  Each stage reads from the previous stage's output directory
and writes its own, enforcing a standardised FASTA / metadata format.

Usage
-----
    genomefactory-cli collect pipeline_config.yaml
"""

import os
import shutil
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class PipelineStage(ABC):
    """Base class that every pipeline stage must implement."""

    @abstractmethod
    def process(self, input_dir: str, output_dir: str, config: dict) -> None:
        """
        Run this stage.

        Parameters
        ----------
        input_dir : str
            Directory containing input files (from previous stage or raw data).
        output_dir : str
            Directory where this stage should write its outputs.
        config : dict
            Stage-specific configuration from the YAML file.
        """
        ...


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

STAGE_REGISTRY: dict = {}


def register_stage(name: str):
    """Decorator that registers a PipelineStage subclass."""
    def decorator(cls):
        STAGE_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Orchestrates a sequence of PipelineStage instances."""

    def __init__(self, pipeline_config: dict):
        self.work_dir = Path(pipeline_config.get("work_dir", "./pipeline_output"))
        self.input_dir = pipeline_config.get("input_dir", None)
        self.stages_cfg = pipeline_config.get("stages", [])

        if not self.stages_cfg:
            raise ValueError("Pipeline config must define at least one stage.")

    # ------------------------------------------------------------------
    def run(self) -> str:
        """Execute every stage in order, chaining directories.

        Returns the path to the final stage's output directory.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)

        prev_output = self.input_dir  # may be None for first stage

        for idx, stage_cfg in enumerate(self.stages_cfg):
            stage_name = stage_cfg.get("name", f"stage_{idx}")
            stage_type = stage_cfg.get("type")
            stage_params = stage_cfg.get("config", {})

            if stage_type not in STAGE_REGISTRY:
                raise ValueError(
                    f"Unknown stage type '{stage_type}'. "
                    f"Available: {list(STAGE_REGISTRY.keys())}"
                )

            stage_cls = STAGE_REGISTRY[stage_type]
            stage_obj = stage_cls()

            stage_out = str(self.work_dir / f"{idx:02d}_{stage_name}")
            os.makedirs(stage_out, exist_ok=True)

            in_dir = prev_output if prev_output else stage_out
            logger.info(
                "[Pipeline] Stage %d/%d: %s (%s)  input=%s  output=%s",
                idx + 1, len(self.stages_cfg), stage_name, stage_type,
                in_dir, stage_out,
            )
            print(
                f"[Pipeline] Stage {idx+1}/{len(self.stages_cfg)}: "
                f"{stage_name} ({stage_type})  "
                f"input={in_dir}  output={stage_out}"
            )

            stage_obj.process(in_dir, stage_out, stage_params)

            prev_output = stage_out

        print(f"[Pipeline] All {len(self.stages_cfg)} stages complete. "
              f"Final output: {prev_output}")
        return prev_output
