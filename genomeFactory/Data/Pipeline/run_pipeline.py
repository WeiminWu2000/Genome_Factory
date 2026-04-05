"""
CLI entry point for the multi-stage bioinformatics pipeline.

    python run_pipeline.py --pipeline_config pipeline.yaml
"""

import argparse
import yaml
import sys

from genomeFactory.Data.Pipeline.pipeline_runner import PipelineRunner
# Import stages so that @register_stage decorators run
import genomeFactory.Data.Pipeline.stages  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="Run a multi-stage bioinformatics pipeline."
    )
    parser.add_argument(
        "--pipeline_config", type=str, required=True,
        help="Path to the pipeline YAML configuration file.",
    )
    args = parser.parse_args()

    with open(args.pipeline_config, "r") as f:
        config = yaml.safe_load(f)

    pipeline_cfg = config.get("pipeline", config)
    runner = PipelineRunner(pipeline_cfg)
    runner.run()


if __name__ == "__main__":
    main()
