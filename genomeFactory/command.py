import argparse
import yaml
import os
import subprocess
import sys
import tempfile
import json


# ======================================================================
# Feature 1: Multi-stage bioinformatics pipeline ("collect" command)
# ======================================================================

def run_collect(config_path: str):
    """
    Run a multi-stage bioinformatics pipeline.

    The YAML at *config_path* defines a ``pipeline`` section with a list
    of stages.  Each stage has a ``type`` (e.g. HostFilter, QualityTrim,
    TaxonExtract, SequenceExtract, CustomCommand) and a ``config`` dict.
    """
    pipeline_script = os.path.join(
        os.path.dirname(__file__), "Data/Pipeline/run_pipeline.py"
    )
    cmd = ["python", pipeline_script, "--pipeline_config", config_path]
    print("[genomefactory-cli] Running collect pipeline:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ======================================================================
# Feature 2: Joint optimization of preprocessing + model training
# ======================================================================

def run_train_joint(config: dict):
    """
    Train with joint optimisation: learnable NormalizationLayer +
    composite loss (task + batch-MMD + bio-preservation).
    """
    joint_script = os.path.join(
        os.path.dirname(__file__), "Train/train_scripts/joint/train_joint.py"
    )

    model_name = config.get("model", {}).get("model_name_or_path", "zhihan1996/DNABERT-2-117M")
    joint_cfg = config.get("joint", {})
    train_cfg = config.get("train", {})
    output_cfg = config.get("output", {})
    dataset_cfg = config.get("dataset", {})

    data_paths = dataset_cfg.get("data_path", ["./dataset"])
    if isinstance(data_paths, list):
        data_path = data_paths[0]
    else:
        data_path = data_paths

    classification = train_cfg.get("classification", True)
    regression = train_cfg.get("regression", False)

    cmd = [
        "python", joint_script,
        "--model_name_or_path", model_name,
        "--data_path", data_path,
        "--classification", str(classification),
        "--regression", str(regression),
        "--lambda_batch", str(joint_cfg.get("lambda_batch", 0.1)),
        "--lambda_bio", str(joint_cfg.get("lambda_bio", 0.05)),
        "--norm_hidden_size", str(joint_cfg.get("norm_hidden_size", 128)),
        "--model_max_length", str(train_cfg.get("model_max_length", [512])[0] if isinstance(train_cfg.get("model_max_length", 512), list) else train_cfg.get("model_max_length", 512)),
        "--per_device_train_batch_size", str(train_cfg.get("per_device_train_batch_size", [8])[0] if isinstance(train_cfg.get("per_device_train_batch_size", 8), list) else train_cfg.get("per_device_train_batch_size", 8)),
        "--per_device_eval_batch_size", str(train_cfg.get("per_device_eval_batch_size", [16])[0] if isinstance(train_cfg.get("per_device_eval_batch_size", 16), list) else train_cfg.get("per_device_eval_batch_size", 16)),
        "--learning_rate", str(train_cfg.get("learning_rate", [3e-5])[0] if isinstance(train_cfg.get("learning_rate", 3e-5), list) else train_cfg.get("learning_rate", 3e-5)),
        "--num_train_epochs", str(train_cfg.get("num_train_epochs", [3])[0] if isinstance(train_cfg.get("num_train_epochs", 3), list) else train_cfg.get("num_train_epochs", 3)),
        "--warmup_steps", str(train_cfg.get("warmup_steps", [50])[0] if isinstance(train_cfg.get("warmup_steps", 50), list) else train_cfg.get("warmup_steps", 50)),
        "--logging_steps", str(train_cfg.get("logging_steps", [10])[0] if isinstance(train_cfg.get("logging_steps", 10), list) else train_cfg.get("logging_steps", 10)),
        "--save_steps", str(train_cfg.get("save_steps", [100])[0] if isinstance(train_cfg.get("save_steps", 100), list) else train_cfg.get("save_steps", 100)),
        "--eval_steps", str(train_cfg.get("eval_steps", [100])[0] if isinstance(train_cfg.get("eval_steps", 100), list) else train_cfg.get("eval_steps", 100)),
        "--evaluation_strategy", str(train_cfg.get("evaluation_strategy", ["steps"])[0] if isinstance(train_cfg.get("evaluation_strategy", "steps"), list) else train_cfg.get("evaluation_strategy", "steps")),
        "--output_dir", str(output_cfg.get("output_dir", ["output_joint"])[0] if isinstance(output_cfg.get("output_dir", "output_joint"), list) else output_cfg.get("output_dir", "output_joint")),
        "--overwrite_output_dir", str(output_cfg.get("overwrite_output_dir", True)),
        "--bf16", str(train_cfg.get("bf16", False)),
        "--fp16", str(train_cfg.get("fp16", False)),
        "--run_name", str(train_cfg.get("run_name", ["joint_run"])[0] if isinstance(train_cfg.get("run_name", "joint_run"), list) else train_cfg.get("run_name", "joint_run")),
    ]

    saved_model_dir = train_cfg.get("saved_model_dir")
    if saved_model_dir:
        if isinstance(saved_model_dir, list):
            saved_model_dir = saved_model_dir[0]
        cmd += ["--saved_model_dir", saved_model_dir]

    print("[genomefactory-cli] Running joint training:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ======================================================================
# Feature 3: Multi-task learning
# ======================================================================

def run_train_mtl(config: dict, config_path: str):
    """
    Multi-task learning: multiple tasks share one backbone, each with
    its own prediction head and loss.
    """
    mtl_script = os.path.join(
        os.path.dirname(__file__), "Train/train_scripts/mtl/train_mtl.py"
    )
    cmd = ["python", mtl_script, "--config_path", config_path]
    print("[genomefactory-cli] Running MTL training:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ======================================================================
# Original commands below
# ======================================================================

def run_sae_regression(config: dict):
    """
    Run SAE regression using the specified model and configuration.
    """
    if config.get("setup", {}).get("type", "first_token") == "first_token":
        sae_script = os.path.join(os.path.dirname(__file__), "Interpretation/SAE/dna_sequence_analysis_firsttoken.py")
    else:
        sae_script = os.path.join(os.path.dirname(__file__), "Interpretation/SAE/dna_sequence_analysis.py")
    cmd = [
        "python", sae_script,
        "--csv_path", config.get("setup", {}).get("csv_path", "feature_seqlength.csv"),
        "--sae_checkpoint_path", config.get("setup", {}).get("sae_checkpoint_path", "/projects/p32572/interprot/interprot/dna_results_dim4096_k64_auxk256_/checkpoints/dnabert2_768_sae4096_k64_auxk256_-step=54000-avg_mse_loss=0.04.ckpt"),
        "--output_path", config.get("setup", {}).get("output_path", "ridge_regression_weights.csv"),
    ]
    print("[genomefactory-cli] Running SAE regression command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_sae_train(config: dict):
    """
    Run SAE training using the specified model and configuration.
    """
    sae_script = os.path.join(os.path.dirname(__file__), "Interpretation/SAE/dna_training.py")
    cmd = [
        "python", sae_script,
        "--data-file", config.get("setup", {}).get("data_file", "sequences_sae.txt"),
        "--d-model", config.get("setup", {}).get("d_model", 768),
        "--d-hidden", config.get("setup", {}).get("d_hidden", 4096),
        "--k", config.get("setup", {}).get("k", 64),
        "--auxk", config.get("setup", {}).get("auxk", 256),
        "--max-epochs", config.get("setup", {}).get("max_epochs", 3),
        "--batch-size", config.get("setup", {}).get("batch_size", 48),
        "--dead-steps-threshold", config.get("setup", {}).get("dead_steps_threshold", 2000),
        "--lr", config.get("setup", {}).get("lr", 2e-4),
        "--num-devices", config.get("setup", {}).get("num_devices", 1),
        "--model-name", config.get("setup", {}).get("model_name", "pGenomeOcean/GenomeOcean-4B"),
        "--wandb-project", config.get("setup", {}).get("wandb_project", "pGenomeOcean"),
        "--num-workers", config.get("setup", {}).get("num_workers", None),
        "--model-suffix", config.get("setup", {}).get("model_suffix", ""),
    ]
    print("[genomefactory-cli] Running SAE training command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_protein(config: dict):
    """
    Run protein prediction using the specified model and configuration.
    """
    model_name_or_path = config.get("model", {}).get("model_name_or_path", "Evo")
    if model_name_or_path.lower() == "evo":
        protein_script = os.path.join(os.path.dirname(__file__), "Inference/protein_generation/autocomplete_structure_Evo.py")
    elif model_name_or_path.lower() == "genomeocean":
        protein_script = os.path.join(os.path.dirname(__file__), "Inference/protein_generation/autocomplete_structure_GO.py")

    
    gen_id = config.get("setup", {}).get("gen_id", "NZ_JAYXHC010000003.1")
    start = config.get("setup", {}).get("start", 157)
    end = config.get("setup", {}).get("end", 1698)
    strand = config.get("setup", {}).get("strand", -1)
    prompt_start = config.get("setup", {}).get("prompt_start", 0)
    prompt_end = config.get("setup", {}).get("prompt_end", 600)
    structure_start = config.get("setup", {}).get("structure_start", 150)
    structure_end = config.get("setup", {}).get("structure_end", 500)
    num = config.get("setup", {}).get("num", 100)
    min_seq_len = config.get("setup", {}).get("min_seq_len", 1000)
    max_seq_len = config.get("setup", {}).get("max_seq_len", 1200)
    foldmason_path = config.get("setup", {}).get("foldmason_path", "/home/zpt6685/foldmason/bin/foldmason")
    output_prefix = config.get("setup", {}).get("output_prefix", "outputs_Evo/gmp")
    
    cmd = [
        "python", protein_script,
        "--gen_id", gen_id,
        "--start", str(start),
        "--end", str(end),
        "--strand", str(strand),
        "--prompt_start", str(prompt_start),
        "--prompt_end", str(prompt_end),
        "--structure_start", str(structure_start),
        "--structure_end", str(structure_end),
        "--num", str(num),
        "--min_seq_len", str(min_seq_len),
        "--max_seq_len", str(max_seq_len),
        "--foldmason_path", foldmason_path,
        "--output_prefix", output_prefix
    ]
    print("[genomefactory-cli] Running protein command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    
def run_process(config: dict):
    """
    Process the downloaded genome files.
    """
    type = config.get("setup", {}).get("type", "normal")
    process_script = os.path.join(os.path.dirname(__file__), "Data/Process/process_all.py")
    if type == "normal":
        root_dir = config.get("setup", {}).get("root_dir", "data/genomes")
        output_dir = config.get("setup", {}).get("output_dir", "data/processed")
        segments_per_species = config.get("setup", {}).get("segments_per_species", 100)
        segment_length = config.get("setup", {}).get("segment_length", 10000)
        train_ratio = config.get("setup", {}).get("train_ratio", 0.7)
        dev_ratio = config.get("setup", {}).get("dev_ratio", 0.15)
        test_ratio = config.get("setup", {}).get("test_ratio", 0.15)
        cmd = [
            "python", process_script,
            "--type", "normal",
            "--root_dir", root_dir,
            "--output_dir", output_dir,
            "--segments_per_species", str(segments_per_species),
            "--segment_length", str(segment_length),
            "--train_ratio", str(train_ratio),
            "--dev_ratio", str(dev_ratio),
            "--test_ratio", str(test_ratio)
        ]
        print("[genomefactory-cli] Running process command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    elif type == "promoter":
        out_dir = config.get("setup", {}).get("out_dir", "promoter_dataset")
        train_ratio = config.get("setup", {}).get("train_ratio", 0.8)
        val_ratio = config.get("setup", {}).get("val_ratio", 0.1)
        cmd = [
            "python", process_script,
            "--type", "promoter",
            "--out_dir", out_dir,
            "--train_ratio", str(train_ratio),
            "--val_ratio", str(val_ratio)
        ]
        print("[genomefactory-cli] Running process command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    elif type == "emp":
        out_dir = config.get("setup", {}).get("out_dir", "epimark_dataset")
        train_ratio = config.get("setup", {}).get("train_ratio", 0.8)
        val_ratio = config.get("setup", {}).get("val_ratio", 0.1)
        cmd = [
            "python", process_script,
            "--type", "emp",
            "--out_dir", out_dir,
            "--train_ratio", str(train_ratio),
            "--val_ratio", str(val_ratio)
        ]
        print("[genomefactory-cli] Running process command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    elif type == "enhancer":
        out_dir = config.get("setup", {}).get("out_dir", "enhancer_dataset")
        train_ratio = config.get("setup", {}).get("train_ratio", 0.8)
        val_ratio = config.get("setup", {}).get("val_ratio", 0.1)
        cmd = [
            "python", process_script,
            "--type", "enhancer",
            "--out_dir", out_dir,
            "--train_ratio", str(train_ratio),
            "--val_ratio", str(val_ratio)
        ]
        print("[genomefactory-cli] Running process command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

def run_train(config: dict):
    """
    Parse YAML (train_full.yaml / train_lora.yaml) and call Finetune.py for training.
    Handles both full and LoRA training, plus optional W&B usage, and custom saved model dir.
    """
    use_lora = config.get("setup", {}).get("use_lora", False)
    model_name_or_path = config.get("setup", {}).get("model_name_or_path", "facebook/opt-125m")
    if use_lora and "evo" in model_name_or_path:
        cache_dir = config.get("setup", {}).get("cache_dir", "/projects/p32572")
        train_script_path = os.path.join(os.path.dirname(__file__), "Train/workflow/full_and_lora/evo_lora.py")
        lora_target_modules = config.get("setup", {}).get("lora_target_modules", "Wqkv,projections")
        evaluation_strategy = config.get("setup", {}).get("evaluation_strategy", "steps")
        overwrite_output_dir = config.get("setup", {}).get("overwrite_output_dir", True)
        log_level = config.get("setup", {}).get("log_level", "info")
        find_unused_parameters = config.get("setup", {}).get("find_unused_parameters", False)
        data_path = config.get("setup", {}).get("data_path", ["/projects/p32572/All_benchmark/H3","/projects/p32572/All_benchmark/H3K4me1"])
        kmer = config.get("setup", {}).get("kmer", [-1,-1])
        run_name = config.get("setup", {}).get("run_name", ["EVO_H3_seed28","EVO_H3K4me1_seed28"])
        model_max_length = config.get("setup", {}).get("model_max_length", [1000,1000])
        per_device_train_batch_size = config.get("setup", {}).get("per_device_train_batch_size", [16,16])
        per_device_eval_batch_size = config.get("setup", {}).get("per_device_eval_batch_size", [16,16])
        gradient_accumulation_steps = config.get("setup", {}).get("gradient_accumulation_steps", [2,2])
        learning_rate = config.get("setup", {}).get("learning_rate", [3e-4,3e-4])
        num_train_epochs = config.get("setup", {}).get("num_train_epochs", [3,3])
        save_steps = config.get("setup", {}).get("save_steps", [200,200]) 
        output_dir = config.get("setup", {}).get("output_dir", ["output_seed/EVO","output_seed/EVO"])
        eval_steps = config.get("setup", {}).get("eval_steps", [200,200])
        warmup_steps = config.get("setup", {}).get("warmup_steps", [50,50])
        logging_steps = config.get("setup", {}).get("logging_steps", [100000,100000])
        lora_r = config.get("setup", {}).get("lora_r", [8,8])
        lora_alpha = config.get("setup", {}).get("lora_alpha", [32,32])
        lora_dropout = config.get("setup", {}).get("lora_dropout", [0.05,0.05])
        lr_scheduler_type = config.get("setup", {}).get("lr_scheduler_type", "linear")
        warmup_ratio = config.get("setup", {}).get("warmup_ratio", 0.1)
        dataloader_drop_last = config.get("setup", {}).get("dataloader_drop_last", True)
        metric_for_best_model = config.get("setup", {}).get("metric_for_best_model", "matthews_correlation")
        weight_decay = config.get("setup", {}).get("weight_decay", 0.01)
        load_best_model_at_end = config.get("setup", {}).get("load_best_model_at_end", True)
        for data_path ,kmer, run_name, model_max_length, per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, \
            learning_rate, num_train_epochs, save_steps, output_dir, eval_steps, warmup_steps, logging_steps, lora_r, lora_alpha, lora_dropout in zip(
            data_path, kmer, run_name, model_max_length, per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, \
            learning_rate, num_train_epochs, save_steps, output_dir, eval_steps, warmup_steps, logging_steps, lora_r, lora_alpha, lora_dropout):
            cmd = [
                "python", train_script_path,
                "--model_name_or_path", model_name_or_path,
                "--use_lora", str(use_lora),
                "--data_path", data_path,
                "--kmer", str(kmer),
                "--run_name", run_name,
                "--model_max_length", str(model_max_length),
                "--per_device_train_batch_size", str(per_device_train_batch_size),
                "--lora_target_modules", lora_target_modules,
                "--cache_dir", cache_dir,
                "--evaluation_strategy", evaluation_strategy,
                "--per_device_eval_batch_size", str(per_device_eval_batch_size),
                "--gradient_accumulation_steps", str(gradient_accumulation_steps),
                "--learning_rate", str(learning_rate),
                "--num_train_epochs", str(num_train_epochs),
                "--save_steps", str(save_steps),
                "--output_dir", output_dir,
                "--eval_steps", str(eval_steps),
                "--warmup_steps", str(warmup_steps),
                "--logging_steps", str(logging_steps),
                "--lora_r", str(lora_r),
                "--lora_alpha", str(lora_alpha),
                "--lora_dropout", str(lora_dropout),
                "--lr_scheduler_type", lr_scheduler_type,
                "--warmup_ratio", str(warmup_ratio),
                "--dataloader_drop_last", str(dataloader_drop_last),
                "--metric_for_best_model", metric_for_best_model,
                "--weight_decay", str(weight_decay),
                "--load_best_model_at_end", str(load_best_model_at_end),
                "--overwrite_output_dir", str(overwrite_output_dir),
                "--log_level", log_level,
                "--find_unused_parameters", str(find_unused_parameters)
            ]
            print("[genomefactory-cli] Running training command:", " ".join(cmd))
            subprocess.run(cmd, check=True) 
        
    else:
        # Basic fields
        use_flash_attention = config.get("train", {}).get("use_flash_attention", True)
        model_name_or_path = config.get("model", {}).get("model_name_or_path", "facebook/opt-125m")
        finetuning_type = config.get("method", {}).get("finetuning_type", "full").lower()
        use_lora = "False"
        if finetuning_type == "lora":
            use_lora = "True"

        # LoRA configs
        lora_r = config.get("method", {}).get("lora_r", [8])
        lora_alpha = config.get("method", {}).get("lora_alpha", [32])
        lora_dropout = config.get("method", {}).get("lora_dropout", [0.05])
        lora_target_modules = config.get("method", {}).get("lora_target", "Wqkv,dense,gated_layers,wo,classifier")

        # Dataset
        data_path = config.get("dataset", {}).get("data_path", ["./dataset"])

        # Training
        run_name = config.get("train", {}).get("run_name", ["run"])
        model_max_length = config.get("train", {}).get("model_max_length", [512])
        per_device_train_batch_size = config.get("train", {}).get("per_device_train_batch_size", [1])
        per_device_eval_batch_size = config.get("train", {}).get("per_device_eval_batch_size", [1])
        gradient_accumulation_steps = config.get("train", {}).get("gradient_accumulation_steps", [1])
        learning_rate = config.get("train", {}).get("learning_rate", [1e-4])
        num_train_epochs = config.get("train", {}).get("num_train_epochs", [1])
        lr_scheduler_type = config.get("train", {}).get("lr_scheduler_type", ["cosine"])
        warmup_ratio = config.get("train", {}).get("warmup_ratio", [0.1])
        classification = config.get("train", {}).get("classification", True)
        regression = config.get("train", {}).get("regression", False)
        fp16 = config.get("train", {}).get("fp16", False)
        bf16 = config.get("train", {}).get("bf16", False)
        ddp_timeout = config.get("train", {}).get("ddp_timeout", 180000000)
        logging_steps = config.get("train", {}).get("logging_steps", [100])
        save_steps = config.get("train", {}).get("save_steps", [100])
        evaluation_strategy = config.get("train", {}).get("evaluation_strategy", ["steps"])
        eval_steps = config.get("train", {}).get("eval_steps", [100])
        warmup_steps = config.get("train", {}).get("warmup_steps", [50])
        output_dir = config.get("output", {}).get("output_dir", ["output"])
        save_total_limit = config.get("train", {}).get("save_total_limit", [3])
        load_best_model_at_end = config.get("train", {}).get("load_best_model_at_end", True)
        overwrite_output_dir = config.get("output", {}).get("overwrite_output_dir", True)

        # Optional custom saved model dir
        saved_model_dir = config.get("train", {}).get("saved_model_dir", None)

        # W&B usage
        use_wandb = config.get("train", {}).get("use_wandb", False)
        if not use_wandb:
            os.environ["WANDB_DISABLED"] = "true"
        else:
            if "WANDB_DISABLED" in os.environ:
                del os.environ["WANDB_DISABLED"]

        finetune_script = os.path.join(os.path.dirname(__file__), "Train/finetune.py")
        if use_lora == "True":
            for lora_r, lora_alpha, lora_dropout, data_path, output_dir, saved_model_dir, run_name, \
            model_max_length, per_device_train_batch_size, per_device_eval_batch_size, \
            gradient_accumulation_steps, learning_rate, num_train_epochs, lr_scheduler_type, \
            warmup_ratio, logging_steps, save_steps, evaluation_strategy, eval_steps, \
            warmup_steps, save_total_limit in zip(
                lora_r, lora_alpha, lora_dropout, data_path, output_dir, saved_model_dir, run_name,
                model_max_length, per_device_train_batch_size, per_device_eval_batch_size,
                gradient_accumulation_steps, learning_rate, num_train_epochs, lr_scheduler_type,
                warmup_ratio, logging_steps, save_steps, evaluation_strategy, eval_steps,
                warmup_steps, save_total_limit):
                cmd = [
                    "python", finetune_script,
                    "--model_name", model_name_or_path,
                    "--use_lora", use_lora,
                    "--finetuning_type", finetuning_type,
                    "--use_flash_attention", str(use_flash_attention),
                    "--lora_r", str(lora_r),
                    "--classification", str(classification),
                    "--regression", str(regression),
                    "--lora_alpha", str(lora_alpha),
                    "--lora_dropout", str(lora_dropout),
                    "--lora_target_modules", str(lora_target_modules),
                    "--data_path", data_path,
                    "--run_name", run_name,
                    "--model_max_length", str(model_max_length),
                    "--per_device_train_batch_size", str(per_device_train_batch_size),
                    "--per_device_eval_batch_size", str(per_device_eval_batch_size),
                    "--gradient_accumulation_steps", str(gradient_accumulation_steps),
                    "--learning_rate", str(learning_rate),
                    "--num_train_epochs", str(num_train_epochs),
                    "--lr_scheduler_type", lr_scheduler_type,
                    "--warmup_ratio", str(warmup_ratio),
                    "--fp16", str(fp16),
                    "--bf16", str(bf16),
                    "--ddp_timeout", str(ddp_timeout),
                    "--logging_steps", str(logging_steps),
                    "--save_steps", str(save_steps),
                    "--evaluation_strategy", evaluation_strategy,
                    "--eval_steps", str(eval_steps),
                    "--warmup_steps", str(warmup_steps),
                    "--output_dir", output_dir,
                    "--save_total_limit", str(save_total_limit),
                    "--load_best_model_at_end", str(load_best_model_at_end),
                    "--overwrite_output_dir", str(overwrite_output_dir),
                ]

                if saved_model_dir:
                    cmd += ["--saved_model_dir", saved_model_dir]

                print("[genomefactory-cli] Running training command:", " ".join(cmd))
                subprocess.run(cmd, check=True)
        else:
            for  data_path, output_dir, saved_model_dir, run_name, \
            model_max_length, per_device_train_batch_size, per_device_eval_batch_size, \
            gradient_accumulation_steps, learning_rate, num_train_epochs, lr_scheduler_type, \
            warmup_ratio, logging_steps, save_steps, evaluation_strategy, eval_steps, \
            warmup_steps, save_total_limit in zip(
                data_path, output_dir, saved_model_dir, run_name,
                model_max_length, per_device_train_batch_size, per_device_eval_batch_size,
                gradient_accumulation_steps, learning_rate, num_train_epochs, lr_scheduler_type,
                warmup_ratio, logging_steps, save_steps, evaluation_strategy, eval_steps,
                warmup_steps, save_total_limit):
                cmd = [
                    "python", finetune_script,
                    "--model_name", model_name_or_path,
                    "--use_lora", use_lora,
                    "--finetuning_type", finetuning_type,
                    "--use_flash_attention", str(use_flash_attention),
                    "--classification", str(classification),
                    "--regression", str(regression),
                    "--data_path", data_path,
                    "--run_name", run_name,
                    "--model_max_length", str(model_max_length),
                    "--per_device_train_batch_size", str(per_device_train_batch_size),
                    "--per_device_eval_batch_size", str(per_device_eval_batch_size),
                    "--gradient_accumulation_steps", str(gradient_accumulation_steps),
                    "--learning_rate", str(learning_rate),
                    "--num_train_epochs", str(num_train_epochs),
                    "--lr_scheduler_type", lr_scheduler_type,
                    "--warmup_ratio", str(warmup_ratio),
                    "--fp16", str(fp16),
                    "--bf16", str(bf16),
                    "--ddp_timeout", str(ddp_timeout),
                    "--logging_steps", str(logging_steps),
                    "--save_steps", str(save_steps),
                    "--evaluation_strategy", evaluation_strategy,
                    "--eval_steps", str(eval_steps),
                    "--warmup_steps", str(warmup_steps),
                    "--output_dir", output_dir,
                    "--save_total_limit", str(save_total_limit),
                    "--load_best_model_at_end", str(load_best_model_at_end),
                    "--overwrite_output_dir", str(overwrite_output_dir),
                ]

                if saved_model_dir:
                    cmd += ["--saved_model_dir", saved_model_dir]

                print("[genomefactory-cli] Running training command:", " ".join(cmd))
                subprocess.run(cmd, check=True)


def run_inference(config: dict):
    """
    Single script for inference: inference.py
    We read model_path from either config['inference']['model_path'] or fallback to config['model']['model_name_or_path'].
    We read dna from config['inference']['dna'], or fallback to a default.
    Then we call 'inference.py'.
    """
    extract = config.get("method", {}).get("extract", False)
    generation = config.get("method", {}).get("generation", False)
    model_path = config.get("model", {}).get("model_name_or_path", "pGenomeOcean/GenomeOcean-100M")
    if generation:
        if "evo" not in model_path:
            dna = config.get("inference", {}).get("dna", "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT")
            min_new_tokens = config.get("method", {}).get("min_new_tokens", 10)
            max_new_tokens = config.get("method", {}).get("max_new_tokens", 10)
            do_sample = config.get("method", {}).get("do_sample", True)
            top_p = config.get("method", {}).get("top_p", 0.9)
            temperature = config.get("method", {}).get("temperature", 1.0)
            num_return_sequences = config.get("method", {}).get("num_return_sequences", 1)
            inference_script = os.path.join(os.path.dirname(__file__), "Inference/generation/genome_ocean_generation.py")
            cmd = [
                "python", inference_script,
                "--model_path", model_path,
                "--dna", dna,
                "--min_new_tokens", str(min_new_tokens),
                "--max_new_tokens", str(max_new_tokens),
                "--do_sample", str(do_sample),
                "--top_p", str(top_p),
                "--temperature", str(temperature),
                "--num_return_sequences", str(num_return_sequences)
            ]
        else:
            dna = config.get("inference", {}).get("dna", "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT")
            n_samples = config.get("method", {}).get("n_samples", 1)
            n_tokens = config.get("method", {}).get("n_tokens", 100)
            temperature = config.get("method", {}).get("temperature", 1.0)
            top_k = config.get("method", {}).get("top_k", 4)
            top_p = config.get("method", {}).get("top_p", 1.0)
            device = config.get("method", {}).get("device", "cuda:0")
            verbose = config.get("method", {}).get("verbose", 1)
            inference_script = os.path.join(os.path.dirname(__file__), "Inference/generation/evo_generation.py")
            cmd = [
                "python", inference_script,
                "--model_path", model_path,
                "--dna", dna,
                "--n_samples", str(n_samples),
                "--n_tokens", str(n_tokens),
                "--temperature", str(temperature),
                "--top_k", str(top_k),
                "--top_p", str(top_p),
                "--device", device,
                "--verbose", str(verbose)
            ]
        print("[genomefactory-cli] Running inference command:", " ".join(cmd))
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    elif extract:
        model_path= config.get("model", {}).get("model_name_or_path", "pGenomeOcean/GenomeOcean-100M")
        
        if "evo" in model_path:
            dna = config.get("inference", {}).get("dna",
            "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT"
            )
            output_file = config.get("inference", {}).get("output_dir", "./embeddings.npy")
            inference_script = os.path.join(
                os.path.dirname(__file__),
                "Inference/extract_embedding/evo_extract.py"
            )
            cmd = [
                "python", inference_script,
                "--model_path", model_path,
                "--dna", dna,
                "--output_file", output_file
            ]
        else:
            dna_list = config.get("inference", {}).get("dna", [
            "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT"
            ])
            output_file = config.get("inference", {}).get("output_dir", "./embeddings.npy")
            inference_script = os.path.join(
                os.path.dirname(__file__),
                "Inference/extract_embedding/extract.py"
            )

        # Build command: one --dna, then all sequences, then --output_file
            cmd = [
                "python", inference_script,
                "--model_path", model_path,
                "--dna",
            ] + dna_list + [
                "--output_file", output_file
            ]

        print("[genomefactory-cli] Running inference command:", " ".join(cmd))
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    else:
        num_labels = config.get("inference", {}).get("num_labels", 1)
        finetuning_type = config.get("method", {}).get("finetuning_type", "full").lower()
        classification = config.get("method", {}).get("classification", True)
        regression = config.get("method", {}).get("regression", False)
        inf_cfg = config.get("inference", {})
        model_path = inf_cfg.get("model_path", "./Trained_model")
        model_max_length = inf_cfg.get("model_max_length", 128)

        dna = inf_cfg.get("dna", "ATTGGTGGAATGCACAGGATATTGTGAAGGAGTACAG...")
        if finetuning_type != "adapter":
            if classification:
                inference_script = os.path.join(os.path.dirname(__file__), "Inference/classification/Inference_classification.py")
            if regression:
                inference_script = os.path.join(os.path.dirname(__file__), "Inference/regression/Inference_regression.py")
            cmd = [
            "python", inference_script,
            "--model_path", model_path,
            "--dna", dna,
            "--model_max_length", str(model_max_length)
            ]
            print("[genomefactory-cli] Running inference command:", " ".join(cmd))
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout
        else:
            if classification:
                inference_script = os.path.join(os.path.dirname(__file__), "Inference/classification/Inference_adapter_classification.py")
            if regression:
                inference_script = os.path.join(os.path.dirname(__file__), "Inference/regression/Inference_adapter_regression.py")
            cmd = [
                "python", inference_script,
                "--model_path", model_path,
                "--dna", dna,
                "--num_labels", str(num_labels),
                "--model_max_length", str(model_max_length)
            ]
            print("[genomefactory-cli] Running inference command:", " ".join(cmd))
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout


def run_webui():
    """
    A Gradio Web UI that includes a 'Download' tab with TWO separate mini-interfaces:
      - By species
      - By link
    Each is shown/hidden based on the radio selection.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please install it via pip install gradio.")
        sys.exit(1)

    ############################################################
    #                  Existing train / inference
    ############################################################
    def on_train_submit(
        use_flash_attention,
        classification,
        regression,
        model_name_or_path,
        finetuning_type,
        use_wandb,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target,
        data_path,
        run_name,
        model_max_length,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        gradient_accumulation_steps,
        learning_rate,
        num_train_epochs,
        lr_scheduler_type,
        warmup_ratio,
        fp16,
        bf16,
        ddp_timeout,
        logging_steps,
        save_steps,
        evaluation_strategy,
        eval_steps,
        warmup_steps,
        output_dir,
        save_total_limit,
        load_best_model_at_end,
        overwrite_output_dir,
        saved_model_dir
    ):
        # Convert comma-separated values to lists
        data_paths = [path.strip() for path in data_path.split(',')]
        output_dirs = [dir.strip() for dir in output_dir.split(',')]
        
        # Handle saved_model_dir
        if saved_model_dir and saved_model_dir.strip():
            saved_model_dirs = [dir.strip() for dir in saved_model_dir.split(',')]
            # Ensure equal length by padding with None if needed
            if len(saved_model_dirs) < len(data_paths):
                saved_model_dirs.extend([None] * (len(data_paths) - len(saved_model_dirs)))
        else:
            saved_model_dirs = [None] * len(data_paths)
            
        # Ensure output_dirs is the same length as data_paths
        if len(output_dirs) < len(data_paths):
            # If there are fewer output directories than data paths, repeat the last one
            output_dirs.extend([output_dirs[-1]] * (len(data_paths) - len(output_dirs)))
        
        # Generic function: Convert comma-separated values to lists
        def parse_to_list(param, param_type=str):
            try:
                # For string parameters, directly use split
                if isinstance(param, str) and ',' in param:
                    return [param_type(val.strip()) for val in param.split(',')]
                # For numeric parameters, convert to string first then split
                elif not isinstance(param, str) and ',' in str(param):
                    return [param_type(val.strip()) for val in str(param).split(',')]
                else:
                    return [param_type(param)]
            except:
                # If conversion fails, return single-value list
                return [param]
        
        # Parse all parameters
        lora_r_list = parse_to_list(lora_r, int)
        lora_alpha_list = parse_to_list(lora_alpha, int)
        lora_dropout_list = parse_to_list(lora_dropout, float)
        
        run_names = parse_to_list(run_name)
        model_max_lengths = parse_to_list(model_max_length, int)
        per_device_train_batch_sizes = parse_to_list(per_device_train_batch_size, int)
        per_device_eval_batch_sizes = parse_to_list(per_device_eval_batch_size, int)
        gradient_accumulation_steps_list = parse_to_list(gradient_accumulation_steps, int)
        learning_rates = parse_to_list(learning_rate, float)
        num_train_epochs_list = parse_to_list(num_train_epochs, int)
        lr_scheduler_types = parse_to_list(lr_scheduler_type)
        warmup_ratios = parse_to_list(warmup_ratio, float)
        logging_steps_list = parse_to_list(logging_steps, int)
        save_steps_list = parse_to_list(save_steps, int)
        evaluation_strategies = parse_to_list(evaluation_strategy)
        eval_steps_list = parse_to_list(eval_steps, int)
        warmup_steps_list = parse_to_list(warmup_steps, int)
        save_total_limits = parse_to_list(save_total_limit, int)
        
        # Ensure all lists match the length of data_paths
        def ensure_length(values):
            if len(values) < len(data_paths):
                # If list is too short, pad with the last value
                values.extend([values[-1]] * (len(data_paths) - len(values)))
            return values[:len(data_paths)]  # Truncate if too long
        
        # Apply length adjustments
        lora_r_list = ensure_length(lora_r_list)
        lora_alpha_list = ensure_length(lora_alpha_list)
        lora_dropout_list = ensure_length(lora_dropout_list)
        run_names = ensure_length(run_names)
        model_max_lengths = ensure_length(model_max_lengths)
        per_device_train_batch_sizes = ensure_length(per_device_train_batch_sizes)
        per_device_eval_batch_sizes = ensure_length(per_device_eval_batch_sizes)
        gradient_accumulation_steps_list = ensure_length(gradient_accumulation_steps_list)
        learning_rates = ensure_length(learning_rates)
        num_train_epochs_list = ensure_length(num_train_epochs_list)
        lr_scheduler_types = ensure_length(lr_scheduler_types)
        warmup_ratios = ensure_length(warmup_ratios)
        logging_steps_list = ensure_length(logging_steps_list)
        save_steps_list = ensure_length(save_steps_list)
        evaluation_strategies = ensure_length(evaluation_strategies)
        eval_steps_list = ensure_length(eval_steps_list)
        warmup_steps_list = ensure_length(warmup_steps_list)
        save_total_limits = ensure_length(save_total_limits)
        
        config = {
            "model": {
                "model_name_or_path": model_name_or_path
            },
            "method": {
                "finetuning_type": finetuning_type,
                "lora_r": lora_r_list,
                "lora_alpha": lora_alpha_list,
                "lora_dropout": lora_dropout_list,
                "lora_target": lora_target,
            },
            "dataset": {
                "data_path": data_paths,
                
            },
            "train": {
                "use_flash_attention": use_flash_attention,
                "run_name": run_names,
                "classification": classification,
                "regression": regression,
                "model_max_length": model_max_lengths,
                "per_device_train_batch_size": per_device_train_batch_sizes,
                "per_device_eval_batch_size": per_device_eval_batch_sizes,
                "gradient_accumulation_steps": gradient_accumulation_steps_list,
                "learning_rate": learning_rates,
                "num_train_epochs": num_train_epochs_list,
                "lr_scheduler_type": lr_scheduler_types,
                "warmup_ratio": warmup_ratios,
                "fp16": fp16,
                "bf16": bf16,
                "ddp_timeout": ddp_timeout,
                "logging_steps": logging_steps_list,
                "save_steps": save_steps_list,
                "evaluation_strategy": evaluation_strategies,
                "eval_steps": eval_steps_list,
                "warmup_steps": warmup_steps_list,
                "save_total_limit": save_total_limits,
                "load_best_model_at_end": load_best_model_at_end,
                "use_wandb": use_wandb,
                "saved_model_dir": saved_model_dirs if any(saved_model_dirs) else None
            },
            "output": {
                "output_dir": output_dirs,
                "overwrite_output_dir": overwrite_output_dir
            }
        }
        import subprocess

        try:
            output = run_train(config)
            return "Training finished successfully!\n"+output
        except subprocess.CalledProcessError as e:
            return f"Training failed: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def run_train(inner_config):
        """
        Just calls the same code as run_train in the CLI scenario.
        But we can replicate or directly call the same function if we like.
        For brevity, we'll call run_train from this file or replicate the logic.
        We'll just replicate the command approach, or do the steps in-memory.
        For clarity, let's replicate the command approach:
        """
        import subprocess
        import os

        # We basically do the same as run_train(config) from above,
        # but in code. We'll do a quick approach: create a temp file, then pass it.
        import tempfile, json

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            yaml_path = tf.name
            import yaml
            yaml.dump(inner_config, tf)
        cmd = f"genomefactory-cli train {yaml_path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(yaml_path)
            # Check if training was actually successful by examining output
            if "error" in out.lower() or "exception" in out.lower() or "failed" in out.lower():
                raise Exception(f"Training failed based on output:\n{out}")
            return out
        except subprocess.CalledProcessError as e:
            os.remove(yaml_path)
            raise e


    def on_inference_submit(model_path, dna, finetuning_type,classification, regression,num_labels,model_max_length):
        config = {
            "method": {
                "finetuning_type": finetuning_type,
                "classification": classification,
                "regression": regression
            },
            "inference": {
                "model_max_length": model_max_length,
                "num_labels": num_labels,
                "model_path": model_path,
                "dna": dna
            } 
        }
        import subprocess, tempfile, json
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            cfg_path = tf.name
            import yaml
            yaml.dump(config, tf)
        cmd = f"genomefactory-cli inference {cfg_path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(cfg_path)
            return f"Inference finished successfully!\n\nOutput:\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(cfg_path)
            return f"Inference failed: {e}\n\nOutput:\n{e.output}"
        except Exception as e:
            os.remove(cfg_path)
            return f"Error: {str(e)}"
    def on_inference_submit_generation(model_path,generation, dna, min_new_tokens, max_new_tokens, do_sample, top_p, temperature, num_return_sequences):
        config = {
            "model": {
                "model_name_or_path": model_path
            },
            "method": {
                "generation": generation,
                "min_new_tokens": min_new_tokens,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "top_p": top_p,
                "temperature": temperature,
                "num_return_sequences": num_return_sequences
            },
            "inference": {
                "dna": dna
            }
        }
        import subprocess, tempfile, json
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            cfg_path = tf.name
            import yaml
            yaml.dump(config, tf)
        cmd = f"genomefactory-cli inference {cfg_path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(cfg_path)
            return f"Inference finished successfully!\n\nOutput:\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(cfg_path)
            return f"Inference failed: {e}\n\nOutput:\n{e.output}"
        except Exception as e:
            os.remove(cfg_path)
            return f"Error: {str(e)}"
        
    def on_inference_submit_generation_evo(model_path, generation, n_samples, n_tokens, temperature, top_k, top_p, device, verbose, gen_dna):
            config = {
                "model": {
                    "model_name_or_path": model_path
                },
                "method": {
                    "generation": generation,
                    "n_samples": n_samples,
                    "n_tokens": n_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "device": device,
                    "verbose": verbose
                },
                "inference": {
                    "dna": gen_dna
                }
            }
            import subprocess, tempfile, json
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
                cfg_path = tf.name
                import yaml
                yaml.dump(config, tf)
            cmd = f"genomefactory-cli inference {cfg_path}"
            try:
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
                os.remove(cfg_path)
                return f"Inference finished successfully!\n\nOutput:\n{out}"
            except subprocess.CalledProcessError as e:
                os.remove(cfg_path)
                return f"Inference failed: {e}\n\nOutput:\n{e.output}"
            except Exception as e:
                os.remove(cfg_path)
                return f"Error: {str(e)}"

    def on_inference_submit_extract(model_path, dna, output_dir, extract):
        if "evo" not in model_path:
            dna=dna.split(",")
        config = {
            "model": {
                "model_name_or_path": model_path
            },
            "method": {
                "extract": extract
            },
            "inference": {
                "dna": dna,
                "output_dir": output_dir
            }
        }
        import subprocess, tempfile, json
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            cfg_path = tf.name
            import yaml
            yaml.dump(config, tf)
        cmd = f"genomefactory-cli inference {cfg_path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(cfg_path)
            return f"Processing finished successfully!\n\nOutput:\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(cfg_path)
            return f"Processing failed: {e}\n\nOutput:\n{e.output}"
        except Exception as e:
            os.remove(cfg_path)
            return f"Error: {str(e)}"
    def on_process_submit(root_dir, output_dir, segments_per_species, segment_length, train_ratio, dev_ratio, test_ratio):
        config = {
            "setup": {
                "root_dir": root_dir,
                "output_dir": output_dir,
                "segments_per_species": segments_per_species,
                "segment_length": segment_length,
                "train_ratio": train_ratio,
                "dev_ratio": dev_ratio,
                "test_ratio": test_ratio
            }
        }
        
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            cfg_path = tf.name
            import yaml
            yaml.dump(config, tf)
            
        cmd = f"genomefactory-cli process {cfg_path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(cfg_path)
            return f"Processing finished successfully!\n\nOutput:\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(cfg_path)
            return f"Processing failed: {e}\n\nOutput:\n{e.output}"
        except Exception as e:
            os.remove(cfg_path)
            return f"Error: {str(e)}"
    ############################################################
    #                  Download Logic
    ############################################################

    def on_download_species(species, folder):
        """
        Invokes the species-based approach:
          genomefactory-cli download (with species in config).
        """
        import subprocess, tempfile
        import yaml

        cfg = {
            "download": {
                "species": species,
                "download_folder": folder if folder.strip() else None
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            path = tf.name
            yaml.dump(cfg, tf)

        cmd = f"genomefactory-cli download {path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(path)
            return f"Species-based download completed!\n\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(path)
            return f"Species-based download failed: {e}\n\nOutput:\n{e.output}"

    def on_download_link(link, folder):
        """
        Invokes direct link approach from CLI.
        """
        import subprocess, tempfile
        import yaml

        cfg = {
            "download": {
                "link": link,
                "download_folder": folder if folder.strip() else None
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            path = tf.name
            yaml.dump(cfg, tf)

        cmd = f"genomefactory-cli download {path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(path)
            return f"Link-based download completed!\n\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(path)
            return f"Link-based download failed: {e}\n\nOutput:\n{e.output}"

    # We'll do some logic to hide / show the species UI vs link UI
    import gradio as gr
    import json

    dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Download/Datasets_species_taxonid_dict.json")
    with open(dict_path, "r", encoding="utf-8") as f:
        species_dict = json.load(f)
    all_species = sorted(list(species_dict.keys()))

    def switch_mode(mode):
        """
        Return instructions to show/hide each panel depending on selection.
        """
        if mode == "By species":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    with gr.Blocks() as demo:
        gr.Markdown("# GenomeFactory Web UI")

        ################################################
        # Train Tab
        ################################################
        with gr.Tab("Train"):
            gr.Markdown("## Training Parameters")
            model_name_or_path = gr.Textbox(value="zhihan1996/DNABERT-2-117M", label="Model Name or Path")
            finetuning_type = gr.Radio(choices=["full", "lora","adapter"], value="full", label="Finetuning Type")
            use_wandb = gr.Checkbox(value=False, label="Use Weights & Biases?")
            use_flash_attention = gr.Checkbox(value=False, label="Use Flash Attention?")
            with gr.Group(visible=False) as lora_group:
                lora_r = gr.Textbox(value="8", label="LoRA r")
                lora_alpha = gr.Textbox(value="32", label="LoRA alpha")
                lora_dropout = gr.Textbox(value="0.05", label="LoRA dropout")
                lora_target = gr.Textbox(value="Wqkv,dense,gated_layers,wo,classifier", label="LoRA target modules")

            def update_lora_visibility(ft_type):
                return gr.update(visible=(ft_type == "lora"))
            
            finetuning_type.change(
                fn=update_lora_visibility,
                inputs=[finetuning_type],
                outputs=[lora_group]
            )

            classification = gr.Checkbox(value=True, label="Classification")
            regression = gr.Checkbox(value=False, label="Regression")
            data_path = gr.Textbox(value="./dataset", label="Data Path(s)", placeholder="Enter comma-separated paths for multiple datasets (e.g., ./dataset1, ./dataset2)")
            run_name = gr.Textbox(value="run", label="Run Name")
            model_max_length = gr.Textbox(value="512", label="Model Max Length")
            per_device_train_batch_size = gr.Textbox(value="1", label="Per device Train Batch Size")
            per_device_eval_batch_size = gr.Textbox(value="1", label="Per device Eval Batch Size")
            gradient_accumulation_steps = gr.Textbox(value="1", label="Gradient Accum Steps")
            learning_rate = gr.Textbox(value="1e-4", label="Learning Rate")
            num_train_epochs = gr.Textbox(value="1", label="Num Train Epochs")
            lr_scheduler_type = gr.Textbox(value="cosine", label="LR Scheduler Type")
            warmup_ratio = gr.Textbox(value="0.1", label="Warmup Ratio")
            fp16 = gr.Checkbox(value=False, label="fp16")
            bf16 = gr.Checkbox(value=False, label="bf16")
            ddp_timeout = gr.Number(value=180000000, label="DDP Timeout")
            logging_steps = gr.Textbox(value="100", label="Logging Steps")
            save_steps = gr.Textbox(value="100", label="Save Steps")
            evaluation_strategy = gr.Textbox(value="steps", label="Evaluation Strategy")
            eval_steps = gr.Textbox(value="100", label="Eval Steps")
            warmup_steps = gr.Textbox(value="50", label="Warmup Steps")
            output_dir = gr.Textbox(value="output", label="Output Dir(s)", placeholder="Enter comma-separated paths for multiple output directories (e.g., output1, output2)")
            save_total_limit = gr.Textbox(value="3", label="Save Total Limit")
            load_best_model_at_end = gr.Checkbox(value=True, label="Load Best Model at End")
            overwrite_output_dir = gr.Checkbox(value=True, label="Overwrite Output Dir")

            saved_model_dir = gr.Textbox(value="", label="Saved Model Dir(s) (optional)", placeholder="Enter comma-separated paths for multiple saved model directories")

            train_button = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Output")

            train_button.click(
                fn=on_train_submit,
                inputs=[
                    use_flash_attention,
                    classification,
                    regression,
                    model_name_or_path,
                    finetuning_type,
                    use_wandb,
                    lora_r,
                    lora_alpha,
                    lora_dropout,
                    lora_target,
                    data_path,
                    
                    run_name,
                    model_max_length,
                    per_device_train_batch_size,
                    per_device_eval_batch_size,
                    gradient_accumulation_steps,
                    learning_rate,
                    num_train_epochs,
                    lr_scheduler_type,
                    warmup_ratio,
                    fp16,
                    bf16,
                    ddp_timeout,
                    logging_steps,
                    save_steps,
                    evaluation_strategy,
                    eval_steps,
                    warmup_steps,
                    output_dir,
                    save_total_limit,
                    load_best_model_at_end,
                    overwrite_output_dir,
                    saved_model_dir
                ],
                outputs=train_output
            )

        ################################################
        # Inference Tab
        ################################################
        with gr.Tab("Inference"):
            gr.Markdown("## Inference Parameters")
            
            # Mode selection
            inf_mode = gr.Radio(choices=["Prediction","Extraction", "GenomeOcean Generation", "Evo Generation"], value="Prediction", label="Inference Mode")
            
            # Prediction part
            with gr.Group(visible=True) as prediction_panel:
                model_max_length = gr.Textbox(value="128", label="Model Max Length")
                finetuning_type = gr.Radio(choices=["full", "lora","adapter"], value="full", label="Finetuning Type")
                classification = gr.Checkbox(value=True, label="Classification")
                regression = gr.Checkbox(value=False, label="Regression")
                num_labels = gr.Number(value=1, label="Number of Labels", visible=False)
                model_path = gr.Textbox(value="./Trained_model", label="Model Path")
                dna = gr.Textbox(value="ATTGGTGGAATGCACAGGATATTGTGAAGGAGTACAG...", label="DNA Sequence")
                inf_button = gr.Button("Start Inference")

                def update_num_labels_visibility(ftype):
                    return gr.update(visible=(ftype == "adapter"))

                finetuning_type.change(
                    fn=update_num_labels_visibility,
                    inputs=[finetuning_type],
                    outputs=[num_labels]
                )
                
            # Generation part
            with gr.Group(visible=False) as generation_panel_genomeocean:
                go_gen_model_path = gr.Textbox(value="pGenomeOcean/GenomeOcean-100M", label="Model Path")
                go_gen_dna = gr.Textbox(value="GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", label="DNA Sequence")
                min_new_tokens = gr.Number(value=10, label="Minimum New Tokens")
                max_new_tokens = gr.Number(value=10, label="Maximum New Tokens")
                do_sample = gr.Checkbox(value=True, label="Do Sample")
                top_p = gr.Number(value=0.9, label="Top P")
                temperature = gr.Number(value=1.0, label="Temperature")
                num_return_sequences = gr.Number(value=1, label="Number of Return Sequences")
                go_generation_mode = gr.Checkbox(value=True, visible=False, label="Generation Mode")
                gen_button_genomeocean = gr.Button("Start Generation")

            with gr.Group(visible=False) as generation_panel_evo:
                evo_gen_model_path = gr.Textbox(value="evo-1-131k-base", label="Model Path")
                evo_gen_dna = gr.Textbox(value="GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", label="DNA Sequence")
                n_samples = gr.Number(value=1, label="Number of Samples")
                n_tokens = gr.Number(value=100, label="Number of Tokens")
                evo_temperature = gr.Number(value=1.0, label="Temperature")
                evo_top_k = gr.Number(value=4, label="Top K")
                evo_top_p = gr.Number(value=1.0, label="Top P")
                device = gr.Textbox(value="cuda:0", label="Device")
                evo_generation_mode = gr.Checkbox(value=True, visible=False, label="Generation Mode")
                verbose = gr.Number(value=1, label="Verbose")
                gen_button_evo = gr.Button("Start Generation")
                
            with gr.Group(visible=False) as extraction_panel:
                extract_model_path = gr.Textbox(value="pGenomeOcean/GenomeOcean-100M", label="Model Path")
                extract_dna = gr.Textbox(value="GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", label="DNA Sequence")
                extract_output_dir = gr.Textbox(value="embeddings.npy", label="Output Directory")
                extraction_mode = gr.Checkbox(value=True, visible=False, label="Extraction Mode")
                extract_button = gr.Button("Start Extraction")
            inf_output = gr.Textbox(label="Inference Output")
            
            # Function to switch between prediction and generation panels
            def switch_inference_mode(mode):
                # The order of returned gr.update calls maps to the 'outputs' of inf_mode.change:
                # [prediction_panel, generation_panel_genomeocean, generation_panel_evo, extraction_panel]
                if mode == "Prediction":
                    # Show Prediction panel, hide others
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                elif mode == "GenomeOcean Generation":
                    # Show GenomeOcean Generation panel, hide others
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                elif mode == "Evo Generation":
                    # Show Evo Generation panel, hide others
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                else:  # This branch handles "Extraction" as per the Radio choices order
                    # Show Extraction panel, hide others
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            
            inf_mode.change(
                fn=switch_inference_mode,
                inputs=[inf_mode],
                outputs=[prediction_panel, generation_panel_genomeocean, generation_panel_evo, extraction_panel]
            )

            # Prediction button click
            inf_button.click(
                fn=on_inference_submit,
                inputs=[model_path, dna, finetuning_type, classification, regression, num_labels, model_max_length],
                outputs=inf_output
            )
            
            # Generation button click
            gen_button_genomeocean.click(
                fn=on_inference_submit_generation,
                inputs=[go_gen_model_path, go_generation_mode, go_gen_dna, min_new_tokens, max_new_tokens, do_sample, top_p, temperature, num_return_sequences],
                outputs=inf_output
            )
            gen_button_evo.click(
                fn=on_inference_submit_generation_evo,
                inputs=[evo_gen_model_path, evo_generation_mode, n_samples, n_tokens, evo_temperature, evo_top_k, evo_top_p, device, verbose, evo_gen_dna],
                outputs=inf_output
            )

            # Extraction button click
            extract_button.click(
                fn=on_inference_submit_extract,
                inputs=[extract_model_path, extract_dna, extract_output_dir, extraction_mode],
                outputs=inf_output
            )
        ################################################
        # Process Tab
        ################################################
        with gr.Tab("Process"):
            gr.Markdown("## Process Genome Data")
            
            root_dir = gr.Textbox(value="data/genomes", label="Root Directory (downloaded genomes)")
            output_dir = gr.Textbox(value="data/processed", label="Output Directory")
            segments_per_species = gr.Number(value=100, label="Segments Per Species")
            segment_length = gr.Number(value=10000, label="Segment Length")
            train_ratio = gr.Number(value=0.7, label="Train Ratio")
            dev_ratio = gr.Number(value=0.15, label="Dev Ratio")
            test_ratio = gr.Number(value=0.15, label="Test Ratio")
            
            process_button = gr.Button("Process Data")
            process_output = gr.Textbox(label="Processing Output")
            
            process_button.click(
                fn=on_process_submit,
                inputs=[
                    root_dir,
                    output_dir,
                    segments_per_species,
                    segment_length,
                    train_ratio,
                    dev_ratio,
                    test_ratio
                ],
                outputs=process_output
            )

        ################################################
        # Download Tab
        ################################################
        with gr.Tab("Download"):
            gr.Markdown("## Download a Genome")

            # A radio to choose approach
            dl_mode = gr.Radio(choices=["By species", "By link"], value="By species", label="Select Download Mode")

            # "By species" UI
            with gr.Group(visible=True) as species_panel:
                species_dropdown = gr.Dropdown(choices=all_species, value="Homo sapiens", label="Species")
                folder_sp = gr.Textbox(value="", label="Download Folder (optional)")
            
            # "By link" UI
            with gr.Group(visible=False) as link_panel:
                link_text = gr.Textbox(value="", label="Direct Link (.fna.gz, etc)")
                folder_link = gr.Textbox(value="", label="Download Folder (optional)")

            dl_button = gr.Button("Download")
            dl_output = gr.Textbox(label="Download Output")

            def switch_mode(mode):
                """Hide or show species_panel vs. link_panel"""
                if mode == "By species":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            dl_mode.change(
                fn=switch_mode,
                inputs=[dl_mode],
                outputs=[species_panel, link_panel]
            )

            # The main download function 
            def dl_unified_fn(mode, sp, folder_s, lk, folder_l):
                if mode == "By species":
                    return on_download_species(sp, folder_s)
                else:
                    return on_download_link(lk, folder_l)

            dl_button.click(
                fn=dl_unified_fn,
                inputs=[dl_mode, species_dropdown, folder_sp, link_text, folder_link],
                outputs=dl_output
            )

    demo.launch(share=True)


def run_download(config_path: str):
    """
    Minimal new function to handle 'download' subcommand from CLI.
    If config has:
      - download.species => do species approach
      - download.link => do link approach
    If both exist, prefer species
    If neither, interactive prompt
    """
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    import yaml
    import json
    import requests, shutil, gzip

    from genomeFactory.Data.Download.GenomeDataset import GenomeDataset  # for species approach

    species = None
    link = None
    download_folder = None

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        species = cfg.get("download", {}).get("species", None)
        link = cfg.get("download", {}).get("link", None)
        download_folder = cfg.get("download", {}).get("download_folder", None)

    if species and link:
        print("Detected both species and link in config. We'll prefer species approach.")
        link = None

    if not species and not link:
        # interactive approach
        print("Download modes:\n1) By species\n2) By link")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            # species approach
            dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Download/Datasets_species_taxonid_dict.json")
            with open(dict_path, "r", encoding="utf-8") as fp:
                species_dict = json.load(fp)
            all_species = sorted(list(species_dict.keys()))
            print("Available species from Datasets_species_taxonid_dict.json:")
            for i, sp in enumerate(all_species):
                print(f"{i+1}. {sp}")
            sp_choice = input("Enter the number or the exact species name: ")
            if sp_choice.isdigit():
                idx = int(sp_choice) - 1
                if 0 <= idx < len(all_species):
                    species = all_species[idx]
                else:
                    print("Invalid choice.")
                    sys.exit(1)
            else:
                if sp_choice in all_species:
                    species = sp_choice
                else:
                    print("Invalid choice.")
                    sys.exit(1)
            folder_choice = input("Enter download folder path (leave empty for default): ")
            if folder_choice.strip():
                download_folder = folder_choice

        elif choice == "2":
            link = input("Paste direct link to .fna(.gz): ").strip()
            folder_c = input("Enter download folder (leave empty for default): ")
            if folder_c.strip():
                download_folder = folder_c
        else:
            print("Invalid choice.")
            sys.exit(1)

    # Actually download
    if species:
        print(f"Downloading by species: {species}")
        if download_folder:
            print(f"Download folder: {download_folder}")
        else:
            print(f"Default folder: ./{species.replace(' ', '_')}")
        try:
            GenomeDataset(species=species, download_folder=download_folder, download=True)
            print("Species-based download completed.")
        except Exception as e:
            print("Error during species-based download:", e)
    elif link:
        if not download_folder:
            download_folder = "./downloaded_genome"
        os.makedirs(download_folder, exist_ok=True)
        filename = link.split("/")[-1]
        local_path = os.path.join(download_folder, filename)

        import requests, shutil, gzip
        print(f"Downloading link: {link}\nStoring to: {local_path}")
        try:
            with requests.get(link, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            if filename.endswith(".gz"):
                # decompress
                decompressed = local_path[:-3]  # remove .gz
                print(f"Decompressing {local_path} -> {decompressed}")
                with gzip.open(local_path, "rb") as f_in, open(decompressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(local_path)  # remove original .gz
                print(f"Link-based download + decompress completed => {decompressed}")
            else:
                print(f"Link-based download completed => {local_path}")
        except Exception as e:
            print("Error during link-based download:", e)




