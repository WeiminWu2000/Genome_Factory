# [Genome-Factory: An Integrated Library for Tuning, Deploying, and Interpreting Genomic Models](http://arxiv.org/abs/2509.12266)

![overview](https://github.com/user-attachments/assets/2f5b2446-2f17-460f-8fc8-e7f529697595)


Genome-Factory is a Python-based integrated library for tuning and deploying genomic foundation models. The framework consists of six components. Genome Collector acquires genomic sequences from public repositories and performs preprocessing (e.g., GC normalization, ambiguous base correction). Model Loader supports major genomic models (e.g., GenomeOcean, EVO, DNABERT-2, HyenaDNA, Caduceus, Nucleotide Transformer) and their tokenizers. Model Trainer configures workflows, adapts models to classification or regression tasks, and executes training with full fine-tuning or parameter-efficient methods (LoRA, adapters). Inference Engine enables embedding extraction and sequence generation. Benchmarker provides standard benchmarks and allows integration of custom evaluation tasks. Biological Interpreter enhances interpretability through sparse auto-encoders.

## Supported Models
The "Variant Type" column specifies how model variants differ: by **parameter size** or by maximum input **sequence length**.

| Model Name             | Variant Type    | Variants                                   |
| ---------------------- | --------------- | ------------------------------------------------ |
| GenomeOcean            | Parameter Size  | 100M / 500M / 4B                                 |
| EVO                    | Sequence Length | 8K / 131K                                        |
| DNABERT-2              | Parameter Size            | 117M                                             |
| Hyenadna               | Sequence Length | 1K / 16K / 32K / 160K / 450K / 1M              |
| Caduceus               | Sequence Length | 1K / 131K                                        |
| Nucleotide Transformer | Parameter Size  | 50M / 100M / 250M / 500M / 1B / 2.5B             |


## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/xxx/Genome_Factory.git
    cd Genome_Factory
    ```
2.  **Install dependencies:**
    ```bash
    # Install primary Python dependencies from requirements file
    pip install -r requirements.txt

    # Install CUDA Toolkit and Compiler 
    # Ensure you have a compatible NVIDIA driver installed and are in a Conda environment.
    conda install cudatoolkit==11.8 -c nvidia
    conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

    # Install additional dependencies for specific features (e.g., Mamba support, Flash Attention)
    
    pip install mamba-ssm==2.2.2 flash-attn==2.7.2.post1

    # Install NCBI Datasets CLI (required for NCBI data download feature)
    conda install conda-forge::ncbi-datasets-cli

    # Install EVO from source
    git clone https://github.com/evo-design/evo.git
    cd evo
    pip install .
    cd .. 
    # IMPORTANT: Return to the Genome-Factory root directory before the next step

    # Install Genome-Factory in editable mode
    pip install -e .
    ```

3.  **Environment Notes:**
    *   For **Genome Ocean**, use `transformers==4.44.2`.
    *   For **other models**, use `transformers==4.29.2`.
    *   For **DNABERT-2**, ensure `triton` is uninstalled: `pip uninstall triton`.

## Usage via CLI (`genomefactory-cli`)

Genome-Factory uses YAML configuration files to define tasks. Example files are provided in `genomeFactory/Examples/`. You can customize the parameters within these files, ensuring you maintain the required YAML structure.

### Data Download

Download genomic data from NCBI:

1.  **Using a config file:** Specify download parameters in a YAML file.
   
    **Download by species:**
    ```bash
    genomefactory-cli download genomeFactory/Examples/download_by_species.yaml
    ```
    **Download by Link:**
    ```bash
    genomefactory-cli download genomeFactory/Examples/download_by_link.yaml
    ```
3.  **Interactively:** Run the command without a config file and follow the prompts in the terminal to specify your download criteria (supports both species-based and link-based downloads).
    ```bash
    genomefactory-cli download
    ```
*Note:* The list of species and their taxon IDs used for downloads is stored in `genomeFactory/Data/Download/Datasets_species_taxonid_dict.json`. This file is not exhaustive; you can extend it by adding new species-to-taxonID pairs to download data for other species as needed.

### Data Processing

Genome-Factory provides tools to prepare data for model fine-tuning. This includes processing data downloaded from NCBI or formatting your own custom datasets.

**1. Processing NCBI Data:**

*   Gather data downloaded using the `download` command into a single folder.
*   Run the processing command with a config file:
    ```bash
    genomefactory-cli process genomeFactory/Examples/process_normal.yaml
    ```
*   The processed data will be ready for input into the model for fine-tuning.

**2. Preparing Custom Datasets:**

If you have your own dataset, format it as follows:

*   Separate your data into three CSV files: `train.csv`, `dev.csv`, and `test.csv`.
*   Each CSV file must have two columns:
    *   The first column should contain the DNA sequences (e.g., `sequence`).
    *   The second column should contain the corresponding labels (e.g., `label`).
        *   For **classification** tasks, labels should be integers (e.g., 0, 1, 2...).
        *   For **regression** tasks, labels should be continuous numbers.
*   Place these three CSV files (`train.csv`, `dev.csv`, `test.csv`) together in a single folder.
*   This folder can then be specified as the input data directory in your training configuration YAML file.

**3. Advanced Processing Features:**

Genome-Factory provides specialized dataset generation tools for common genomic machine learning tasks:

*   Promoter region dataset: Generate promoter vs. non-promoter classification data from the EPDnew database (hg38, mm10, danRer11)
    ```bash
    genomefactory-cli process genomeFactory/Examples/process_promoter.yaml
    ```

*   Epigenetic mark dataset: Create gene body sequences with H3K36me3 signal classification from ENCODE/Roadmap data (hg38, mm10)
    ```bash
    genomefactory-cli process genomeFactory/Examples/process_emp.yaml
    ```

*   Enhancer region dataset: Build enhancer vs. non-enhancer classification data from FANTOM5 annotations (hg38, mm10)
    ```bash
    genomefactory-cli process genomeFactory/Examples/process_enhancer.yaml
    ```

*   All datasets feature quality control, configurable train/val/test splits, and output CSV files with `sequence,label` format.

### Custom Multi-Stage Pipelines (Genome Collector)

Genome-Factory supports user-defined, multi-stage bioinformatics pipelines beyond simple sequence extraction. Users chain pipeline stages in a YAML config, and each stage reads from the previous stage's output directory.

**Built-in stages:**

| Stage | Description |
|-------|-------------|
| `HostFilter` | Remove host-derived reads via minimap2 alignment (requires minimap2 + samtools) |
| `QualityTrim` | Filter by length, GC content, N-content; optional adapter trimming |
| `TaxonExtract` | Keep sequences matching specific taxon IDs or species names |
| `SequenceExtract` | Extract fixed-length segments and produce train/dev/test CSV splits |
| `CustomCommand` | Run an arbitrary shell command with `{input_dir}` / `{output_dir}` placeholders |

**Example pipeline config** (`genomeFactory/Examples/collect_pipeline.yaml`):

```yaml
pipeline:
  work_dir: "./pipeline_output"
  input_dir: "./raw_genomes"
  stages:
    - name: "host_filter"
      type: "HostFilter"
      config:
        host_ref: "/path/to/host_genome.fa"
        threads: 4
    - name: "quality_trim"
      type: "QualityTrim"
      config:
        min_length: 100
        max_n_frac: 0.05
        gc_low: 0.3
        gc_high: 0.7
    - name: "sequence_extract"
      type: "SequenceExtract"
      config:
        segments_per_file: 50
        segment_length: 1000
        train_ratio: 0.7
        dev_ratio: 0.15
```

**Run:**
```bash
genomefactory-cli collect genomeFactory/Examples/collect_pipeline.yaml
```

Users can also implement custom stages by subclassing `PipelineStage` and implementing `process(input_dir, output_dir, config)`. See `genomeFactory/Data/Pipeline/stages.py` for examples.

### Training

For fine-tuning GFMs, Genome-Factory supports two primary task types: **classification** and **regression**. You specify the desired `task_type` in the training YAML configuration file.

Fine-tune GFMs using different methods:

*   **Full Fine-tuning:**
    ```bash
    genomefactory-cli train genomeFactory/Examples/train_full.yaml
    ```
*   **LoRA (Low-Rank Adaptation):**
    ```bash
    genomefactory-cli train genomeFactory/Examples/train_lora.yaml
    ```
    *   Specify target modules in the YAML file:
        *   `all`: Targets all linear layers.
        *   `all_in_and_out_proj`: Targets input/output projection layers and the final classification layer.
        *   *Custom*: Specify module names directly.
    *   For **Evo**:
        ```bash
        genomefactory-cli train genomeFactory/Examples/train_evo_lora.yaml
        ```
*   **Adapter:**
    ```bash
    genomefactory-cli train genomeFactory/Examples/train_adapter.yaml
    ```
    *   Customize the adapter architecture in `genomeFactory/Train/workflow/adapter/adapter_model/Adapter.py` for potentially better performance on specific downstream tasks.

    *Note:* Training settings like batch size, learning rate, and epochs can be customized in the respective YAML files for all methods.

    **Note on Flash Attention:** To enable Flash Attention, set the `flash_attention` argument to `true` in your YAML configuration file. You must also enable mixed-precision training by setting either `bf16: true` or `fp16: true`. If `flash_attention` is set to `false`, or if a specific GFM does not support this argument, the model's default attention mechanism will be used.

    **Benchmarking:** After fine-tuning, performance metrics are saved to a JSON file. You can use these metrics for benchmarking (e.g., comparing the performance of different models or tuning methods on specific tasks).

#### Joint Optimization of Preprocessing and Training

The standard pipeline treats preprocessing (GC normalization, quality filtering) and model training as separate steps, so they cannot inform each other. The `train_joint` command enables joint optimization by:

1. Inserting a **learnable NormalizationLayer** (a lightweight residual MLP) between the tokenizer embeddings and the foundation model encoder. This layer starts as a near-identity function and gradually learns a task-informed normalization during fine-tuning.

2. Training with a **composite loss**:
   - **L_task**: the standard task loss (cross-entropy for classification, MSE for regression)
   - **L_batch** (MMD): Maximum Mean Discrepancy penalty that encourages batch-invariant representations, mitigating technical batch effects
   - **L_bio** (k-mer preservation): penalizes the normalization layer if it distorts the embedding space in a way that loses k-mer frequency structure — sequences with similar biological motif content must remain close in embedding space after normalization

   `L = L_task + λ_batch × L_batch + λ_bio × L_bio`

3. Both the normalization parameters and the model parameters are optimized end-to-end, so data normalization adapts to the final task rather than being decided a priori.

**Example config** (`genomeFactory/Examples/train_joint.yaml`):
```yaml
model:
  model_name_or_path: "zhihan1996/DNABERT-2-117M"
joint:
  lambda_batch: 0.1        # MMD batch-invariance weight
  lambda_bio: 0.05         # k-mer preservation weight
  norm_hidden_size: 128    # NormalizationLayer MLP hidden dim
dataset:
  data_path: ["./my_dataset"]
train:
  classification: true
  num_train_epochs: [3]
  learning_rate: [3.0e-5]
  # ... other training arguments ...
```

**Run:**
```bash
genomefactory-cli train_joint genomeFactory/Examples/train_joint.yaml
```

The trained model and NormalizationLayer weights are saved together. See `genomeFactory/Train/workflow/joint/` for implementation details.

#### Multi-Task Learning (MTL)

Genome-Factory supports simultaneous optimization across multiple related tasks from a single shared backbone. When the training config contains an `mtl` section, the framework automatically:

1. Loads a shared foundation model backbone (e.g., DNABERT-2)
2. Attaches one prediction head per task (classification head with cross-entropy loss, or regression head with MSE loss)
3. Samples mini-batches via round-robin or proportional scheduling across tasks
4. Computes a weighted total loss: `L = Σ(w_i × L_i)` where weights `w_i` are user-configurable

**Example config** (`genomeFactory/Examples/train_mtl.yaml`):
```yaml
model:
  model_name_or_path: "zhihan1996/DNABERT-2-117M"
mtl:
  sampling_strategy: "round_robin"   # or "proportional"
  tasks:
    - name: "histone_classification"
      type: "classification"
      data_path: "./histone_data"
      num_labels: 2
      weight: 1.0
    - name: "expression_regression"
      type: "regression"
      data_path: "./expression_data"
      weight: 0.5
train:
  model_max_length: 128
  per_device_train_batch_size: 8
  learning_rate: 3.0e-5
  num_train_epochs: 3
  # ... other training arguments ...
output:
  output_dir: "output_mtl"
```

**Run:**
```bash
genomefactory-cli train genomeFactory/Examples/train_mtl.yaml
```

After training, per-task evaluation metrics are saved to `output_dir/results/mtl_eval_results.json`, and the shared backbone plus per-task heads are saved separately (`task_heads.pt`). This enables joint representation learning — for example, simultaneously training histone modification prediction (classification) and gene expression regression from the same DNABERT-2 backbone.

### Inference

Use trained models for prediction, generation, or embedding extraction:

1.  **Prediction:** (Predict properties of DNA sequences). Ensure the `task_type` specified in your inference YAML file (`classification` or `regression`) matches the task the model was originally fine-tuned for.
    *   **Full:**
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_full.yaml
        ```
    *   **LoRA:**
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_lora.yaml
        ```
    *   **Adapter:**
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_adapter.yaml
        ```
        *   *Note:* For Adapter-based classification, specify the number of labels (`num_label`) in the YAML. For regression, set `num_label: 1`. Full/LoRA methods infer this automatically.

2.  **Generation:** (Generate new DNA sequences based on existing ones). Applicable to compatible GFMs.
    *   For **GenomeOcean**:
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_generation_genomeocean.yaml
        ```
    *   For **Evo**:
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_generation_evo.yaml
        ```

3.  **Embedding Extraction:** (Extract the last hidden state embeddings from sequences).
    *   General Case:
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_extract.yaml
        ```
    *   For **Evo** specifically:
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_extract_evo.yaml
        ```

4.  **Protein Generation:** (Generate biologically realistic protein sequences with structural constraints via FoldMason integration).

- **Structure-aware generation**: Apply structural constraints during sequence generation
- **Multi-model support**: Evo and GenomeOcean
- **Length control**: Flexible sequence lengths
- **Genomic context**: Condition on specified genomic coordinates
- **Batch processing**: Generate multiple variants

Run:

```bash
genomefactory-cli protein genomeFactory/Examples/protein_generation.yaml
```


### Interpretation

Genome-Factory provides comprehensive tools for understanding and interpreting genomic foundation models through sparse autoencoder (SAE) interpretation to provide deep insights into model behavior and biological significance.

#### Sparse Autoencoder (SAE) Analysis
- ** Latent Feature Discovery**: Identify interpretable features learned by genomic foundation models
- ** Ridge Regression Evaluation**: Quantitative assessment of feature importance for downstream tasks
- ** First-token vs Mean-pooled Analysis**: Compare different pooling strategies for sequence representation
- ** Feature Weight Analysis**: Understand which SAE features contribute most to biological predictions

####  Quick Start Guide

##### SAE-Based Feature Analysis

Complete workflow for SAE training and interpretation:

###### Step 1: Train SAE Model

```bash
genomefactory-cli sae_train genomeFactory/Examples/sae_train.yaml
```

Configure the following parameters in the YAML file:

```yaml
data_file: "<YOUR_SEQUENCE_FILE>"
d_model: <MODEL_DIMENSION>
d_hidden: <HIDDEN_DIMENSION>
batch_size: <BATCH_SIZE>
lr: <LEARNING_RATE>
k: <K_VALUE>
auxk: <AUXK_VALUE>
dead_steps_threshold: <THRESHOLD_STEPS>
max_epochs: <MAX_EPOCHS>
num_devices: <NUM_DEVICES>
model_suffix: "<MODEL_SUFFIX>"
wandb_project: "<PROJECT_NAME>"
num_workers: <NUM_WORKERS>
model_name: "<MODEL_NAME>"
```

###### Step 2: Downstream Evaluations with Ridge Regression

**A. First-token latent embedding analysis:**

```bash
genomefactory-cli sae_regression genomeFactory/Examples/sae_regression.yaml
```

Configure the following parameters in the YAML file:

```yaml
csv_path: "<FEATURE_CSV_PATH>"
sae_checkpoint_path: "<SAE_CHECKPOINT_PATH>"
output_path: "<OUTPUT_CSV_PATH>"
type: "first_token"
```

**B. Mean-pooled latent embedding analysis:**

```bash
genomefactory-cli sae_regression genomeFactory/Examples/sae_regression.yaml
```

Configure the following parameters in the YAML file:

```yaml
csv_path: "<FEATURE_CSV_PATH>"
sae_checkpoint_path: "<SAE_CHECKPOINT_PATH>"
output_path: "<OUTPUT_CSV_PATH>"
type: "mean"
```

## Usage via Web UI

Access all Genome-Factory functionalities through a graphical interface:

```bash
genomefactory-cli webui
```

This command launches a web server. Open the provided URL in your browser to use the WebUI.

## Citation

If you find Genome-Factory useful, we would appreciate it if you consider citing our work:

```
@misc{genomefactory2025,
  title     = {Genome-Factory: An Integrated Library for Tuning, Deploying, and Interpreting Genomic Models},
  author    = {Weimin Wu and Xuefeng Song and Yibo Wen and Qinjie Lin and Zhihan Zhou and Jerry Yao-Chieh Hu and Zhong Wang and Han Liu},
  year      = {2025},
  archivePrefix = {arXiv},
  url       = {http://arxiv.org/abs/2509.12266}
}
```


## Reference

```
LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models.
Zheng, Yaowei, Richong Zhang, Junhao Zhang, YeYanhan YeYanhan, and Zheyan Luo.
In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 400-410. 2024.
```
