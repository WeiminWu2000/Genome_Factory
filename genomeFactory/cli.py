# genomeFactory/cli.py

import argparse
import yaml
import os
import subprocess
import sys

from .command import (
    run_train,
    run_inference,
    run_webui,
    run_download,
    run_process,
    run_sae_train,
    run_sae_regression,
    run_protein,
    run_collect,
    run_train_joint,
)


def main():
    parser = argparse.ArgumentParser(
        description="GenomeFactory Command Line Interface"
    )
    parser.add_argument("command", choices=[
                            "train", "inference", "webui", "download", "process",
                            "sae_train", "sae_regression", "protein",
                            "collect", "train_joint",
                        ],
                        help="Command to run")
    parser.add_argument("config_path", type=str, nargs="?",
                        help="Path to the YAML config file (not required for webui or download)")
    args = parser.parse_args()
    
    if args.command == "webui":
        run_webui()
        return

    if args.command == "download":
        run_download(args.config_path)
        return

    if not args.config_path:
        print("Error: config_path is required for train/inference commands.")
        sys.exit(1)

    # Load YAML config
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.command == "collect":
        run_collect(args.config_path)
        return
    elif args.command == "train_joint":
        run_train_joint(config)
        return

    if args.command == "train":
        # Check if this is a multi-task learning config
        if config.get("mtl"):
            from .command import run_train_mtl
            run_train_mtl(config, args.config_path)
        else:
            run_train(config)
    elif args.command == "inference":
        output = run_inference(config)
        print("Inference output:\n", output)
    elif args.command == "process":
        run_process(config)
    elif args.command == "sae_train":
        run_sae_train(config)
    elif args.command == "sae_regression":
        run_sae_regression(config)
    elif args.command == "protein":
        run_protein(config)

if __name__ == "__main__":
    main()



