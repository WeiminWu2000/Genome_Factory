"""
Built-in pipeline stages for the Genome Collector.

Each stage implements ``PipelineStage.process(input_dir, output_dir, config)``.
Stages read FASTA/FASTQ from *input_dir* and write filtered results to
*output_dir*, so they can be freely chained in any order.
"""

import os
import re
import glob
import random
import shutil
import subprocess
import logging
from pathlib import Path

from .pipeline_runner import PipelineStage, register_stage

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------

def _fasta_records(path: str):
    """Yield (header, sequence) tuples from a FASTA file."""
    header, seq_parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


def _write_fasta(records, out_path: str):
    """Write [(header, sequence), ...] to a FASTA file."""
    with open(out_path, "w") as fh:
        for hdr, seq in records:
            fh.write(f"{hdr}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")


def _find_fasta(directory: str):
    """Return list of FASTA files in *directory*."""
    exts = ("*.fa", "*.fasta", "*.fna", "*.fa.gz", "*.fasta.gz", "*.fna.gz")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, ext)))
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(set(files))


# -----------------------------------------------------------------------
# 1. HostFilter  –  remove host-derived reads via minimap2
# -----------------------------------------------------------------------

@register_stage("HostFilter")
class HostFilterStage(PipelineStage):
    """
    Align reads to a host reference genome with minimap2 and keep only
    *unmapped* reads (i.e. non-host sequences).

    Config keys
    -----------
    host_ref : str   – path to host genome (FASTA or .mmi index)
    threads  : int   – minimap2 threads (default 4)
    """

    def process(self, input_dir, output_dir, config):
        host_ref = config.get("host_ref")
        threads = config.get("threads", 4)

        if not host_ref or not os.path.exists(host_ref):
            raise FileNotFoundError(
                f"HostFilter: host_ref not found: {host_ref}. "
                "Provide a valid host genome FASTA or minimap2 index."
            )

        for fpath in _find_fasta(input_dir):
            basename = os.path.basename(fpath)
            out_path = os.path.join(output_dir, basename)
            sam_tmp = os.path.join(output_dir, basename + ".sam")

            # Align, keep unmapped (-f 4), convert back to FASTA
            cmd_align = (
                f"minimap2 -a -t {threads} {host_ref} {fpath} > {sam_tmp}"
            )
            cmd_extract = (
                f"samtools fasta -f 4 {sam_tmp} > {out_path}"
            )

            print(f"  [HostFilter] Aligning {basename} against host genome ...")
            try:
                subprocess.run(cmd_align, shell=True, check=True)
                subprocess.run(cmd_extract, shell=True, check=True)
            except FileNotFoundError:
                raise RuntimeError(
                    "HostFilter requires 'minimap2' and 'samtools' on PATH."
                )
            finally:
                if os.path.exists(sam_tmp):
                    os.remove(sam_tmp)

            print(f"  [HostFilter] {basename} → {out_path}")


# -----------------------------------------------------------------------
# 2. QualityTrim  –  adapter removal & length / quality filtering
# -----------------------------------------------------------------------

@register_stage("QualityTrim")
class QualityTrimStage(PipelineStage):
    """
    Filter sequences by length, GC content, and ambiguous-base fraction.
    Works on plain FASTA (no quality scores needed).

    Config keys
    -----------
    min_length   : int   – minimum sequence length (default 100)
    max_length   : int   – maximum sequence length (default 1000000)
    max_n_frac   : float – max fraction of N bases (default 0.05)
    gc_low       : float – min GC content (default 0.0)
    gc_high      : float – max GC content (default 1.0)
    adapter_seq  : str   – adapter to trim from 3' end (optional)
    """

    def process(self, input_dir, output_dir, config):
        min_len = config.get("min_length", 100)
        max_len = config.get("max_length", 1_000_000)
        max_n = config.get("max_n_frac", 0.05)
        gc_lo = config.get("gc_low", 0.0)
        gc_hi = config.get("gc_high", 1.0)
        adapter = config.get("adapter_seq", "")

        for fpath in _find_fasta(input_dir):
            basename = os.path.basename(fpath)
            kept = []
            total = 0
            for hdr, seq in _fasta_records(fpath):
                total += 1
                seq = seq.upper()

                # Trim adapter from 3' end
                if adapter:
                    idx = seq.rfind(adapter)
                    if idx >= 0:
                        seq = seq[:idx]

                # Length filter
                if not (min_len <= len(seq) <= max_len):
                    continue

                # N-content filter
                if len(seq) > 0 and seq.count("N") / len(seq) > max_n:
                    continue

                # GC filter
                gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
                if not (gc_lo <= gc <= gc_hi):
                    continue

                kept.append((hdr, seq))

            out_path = os.path.join(output_dir, basename)
            _write_fasta(kept, out_path)
            print(f"  [QualityTrim] {basename}: {total} → {len(kept)} sequences")


# -----------------------------------------------------------------------
# 3. TaxonExtract  –  keep sequences matching specific taxon IDs
# -----------------------------------------------------------------------

@register_stage("TaxonExtract")
class TaxonExtractStage(PipelineStage):
    """
    Keep FASTA records whose header contains one of the given taxon IDs
    or taxon names.  This is a header-based filter (fast, no external tool
    needed).  For full taxonomic classification use Kraken2 externally and
    feed the classified output into a subsequent stage.

    Config keys
    -----------
    taxon_ids    : list[int]  – NCBI taxon IDs to keep
    taxon_names  : list[str]  – species / taxon name substrings to keep
    """

    def process(self, input_dir, output_dir, config):
        taxon_ids = [str(t) for t in config.get("taxon_ids", [])]
        taxon_names = [n.lower() for n in config.get("taxon_names", [])]

        if not taxon_ids and not taxon_names:
            raise ValueError(
                "TaxonExtract: provide at least one of taxon_ids or taxon_names."
            )

        patterns = taxon_ids + taxon_names

        for fpath in _find_fasta(input_dir):
            basename = os.path.basename(fpath)
            kept, total = [], 0
            for hdr, seq in _fasta_records(fpath):
                total += 1
                hdr_lower = hdr.lower()
                if any(p in hdr_lower for p in patterns):
                    kept.append((hdr, seq))

            out_path = os.path.join(output_dir, basename)
            _write_fasta(kept, out_path)
            print(f"  [TaxonExtract] {basename}: {total} → {len(kept)} sequences")


# -----------------------------------------------------------------------
# 4. SequenceExtract  –  extract fixed-length segments for ML training
# -----------------------------------------------------------------------

@register_stage("SequenceExtract")
class SequenceExtractStage(PipelineStage):
    """
    Extract random fixed-length segments from genome FASTA files and
    produce train / dev / test CSV splits ready for the GenomeFactory
    training pipeline.

    Config keys
    -----------
    segments_per_file : int   – segments to extract per FASTA file (default 100)
    segment_length    : int   – length of each segment (default 1000)
    train_ratio       : float – (default 0.7)
    dev_ratio         : float – (default 0.15)
    seed              : int   – random seed (default 42)
    """

    def process(self, input_dir, output_dir, config):
        segs_per = config.get("segments_per_file", 100)
        seg_len = config.get("segment_length", 1000)
        train_r = config.get("train_ratio", 0.7)
        dev_r = config.get("dev_ratio", 0.15)
        seed = config.get("seed", 42)

        random.seed(seed)
        all_segments = []  # (sequence, label)

        fasta_files = _find_fasta(input_dir)
        if not fasta_files:
            print(f"  [SequenceExtract] No FASTA files found in {input_dir}")
            return

        for label, fpath in enumerate(fasta_files):
            seqs = []
            for _, seq in _fasta_records(fpath):
                seq = seq.upper()
                seqs.append(seq)

            full_genome = "".join(seqs)
            if len(full_genome) < seg_len:
                print(f"  [SequenceExtract] {fpath}: genome too short, skipping")
                continue

            count = 0
            for _ in range(segs_per * 10):  # try up to 10x
                if count >= segs_per:
                    break
                start = random.randint(0, len(full_genome) - seg_len)
                segment = full_genome[start:start + seg_len]
                # Skip segments with too many N bases
                if segment.count("N") / seg_len > 0.05:
                    continue
                all_segments.append((segment, label))
                count += 1

            print(f"  [SequenceExtract] {os.path.basename(fpath)}: "
                  f"extracted {count} segments (label={label})")

        random.shuffle(all_segments)
        n = len(all_segments)
        n_train = int(n * train_r)
        n_dev = int(n * dev_r)

        splits = {
            "train.csv": all_segments[:n_train],
            "dev.csv": all_segments[n_train:n_train + n_dev],
            "test.csv": all_segments[n_train + n_dev:],
        }

        for fname, data in splits.items():
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w") as fh:
                fh.write("sequence,label\n")
                for seq, lbl in data:
                    fh.write(f"{seq},{lbl}\n")
            print(f"  [SequenceExtract] {fname}: {len(data)} samples")


# -----------------------------------------------------------------------
# 5. CustomStage  –  run an arbitrary shell command
# -----------------------------------------------------------------------

@register_stage("CustomCommand")
class CustomCommandStage(PipelineStage):
    """
    Run a user-provided shell command.  The placeholders ``{input_dir}``
    and ``{output_dir}`` in the command string are replaced at runtime.

    Config keys
    -----------
    command : str  – shell command template
    """

    def process(self, input_dir, output_dir, config):
        cmd_template = config.get("command", "")
        if not cmd_template:
            raise ValueError("CustomCommand: 'command' not specified in config.")

        cmd = cmd_template.replace("{input_dir}", input_dir).replace(
            "{output_dir}", output_dir
        )
        print(f"  [CustomCommand] Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
