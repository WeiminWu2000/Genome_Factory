"""
Auxiliary loss components for joint optimization of preprocessing and
model training.

- **L_batch** (MMD): Maximum Mean Discrepancy to encourage batch-
  invariant representations.
- **L_bio** (k-mer preservation): penalises the normalisation layer if
  it distorts the embedding space in a way that loses k-mer frequency
  structure — a direct proxy for biological motif content.
"""

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------
# L_batch: MMD batch-invariance loss
# -----------------------------------------------------------------------

def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0):
    """Gaussian RBF kernel between row vectors of x and y."""
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)
    dist = x_sq + y_sq.T - 2.0 * x @ y.T
    return torch.exp(-dist / (2.0 * sigma ** 2))


def compute_mmd_loss(embeddings: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute an MMD penalty by splitting the batch in half and measuring
    distribution distance.  This does not require explicit batch-origin
    labels — it encourages all samples to live in a common
    representation space.

    Parameters
    ----------
    embeddings : (B, D) tensor  –  pooled embeddings for the batch.
    sigma      : float          –  RBF kernel bandwidth.

    Returns
    -------
    Scalar MMD^2 loss.
    """
    n = embeddings.size(0)
    if n < 4:
        return torch.tensor(0.0, device=embeddings.device)

    half = n // 2
    x = embeddings[:half]
    y = embeddings[half: 2 * half]

    kxx = _gaussian_kernel(x, x, sigma).mean()
    kyy = _gaussian_kernel(y, y, sigma).mean()
    kxy = _gaussian_kernel(x, y, sigma).mean()

    mmd2 = kxx + kyy - 2.0 * kxy
    return mmd2.clamp(min=0.0)


# -----------------------------------------------------------------------
# L_bio: k-mer frequency preservation loss
# -----------------------------------------------------------------------

def _count_kmers(input_ids: torch.Tensor, vocab_size: int, k: int = 3) -> torch.Tensor:
    """
    Compute k-mer frequency vectors from token ID sequences.

    For each sequence in the batch, count how often each k-mer
    (consecutive k tokens) appears.  The result is a normalised
    frequency distribution per sequence — this captures motif content
    and compositional bias directly from the DNA tokens.

    Parameters
    ----------
    input_ids  : (B, L) long tensor of token IDs.
    vocab_size : int — tokenizer vocabulary size.
    k          : int — k-mer size (default 3, i.e. tri-nucleotide).

    Returns
    -------
    (B, vocab_size^k) float tensor of normalised k-mer frequencies.
    Capped at vocab_size^k = min(vocab_size^k, 4096) to limit memory.
    """
    B, L = input_ids.shape
    if L < k:
        n_bins = min(vocab_size ** k, 4096)
        return torch.zeros(B, n_bins, device=input_ids.device)

    # Encode each k-mer as a single integer: id[0]*V^(k-1) + id[1]*V^(k-2) + ...
    n_bins = min(vocab_size ** k, 4096)

    # Sliding window of k consecutive tokens
    # Shape: (B, L-k+1, k)
    windows = input_ids.unfold(dimension=1, size=k, step=1)  # (B, L-k+1, k)

    # Compute k-mer index via polynomial hashing
    multipliers = torch.tensor(
        [vocab_size ** (k - 1 - i) for i in range(k)],
        device=input_ids.device, dtype=torch.long,
    )
    kmer_ids = (windows * multipliers.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, L-k+1)

    # Clamp to valid bin range
    kmer_ids = kmer_ids.clamp(0, n_bins - 1)

    # Count frequencies
    freq = torch.zeros(B, n_bins, device=input_ids.device)
    for b in range(B):
        freq[b].scatter_add_(0, kmer_ids[b], torch.ones_like(kmer_ids[b], dtype=torch.float))

    # Normalise to probability distribution
    freq = freq / freq.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return freq


def compute_bio_preservation_loss(
    input_ids: torch.Tensor,
    original_embeds: torch.Tensor,
    normalized_embeds: torch.Tensor,
    vocab_size: int,
    k: int = 3,
) -> torch.Tensor:
    """
    K-mer frequency preservation loss.

    Measures whether the normalised embeddings can still reconstruct the
    original k-mer frequency structure of each sequence.  Concretely:

    1. Compute the ground-truth k-mer frequency vector from ``input_ids``
       (this is the real biological signal — motif content, GC composition,
       dinucleotide bias, etc.).
    2. Project the mean-pooled normalised embeddings down to predict these
       frequencies via a simple linear projection (no learnable params —
       we use the pseudo-inverse so the loss reflects information content,
       not a learned decoder).
    3. Loss = KL divergence between true frequencies and predicted ones.

    If the normalisation layer destroys k-mer structure, the normalised
    embeddings will be unable to reconstruct these frequencies and the
    loss increases.  This directly penalises losing biological signal.

    As a simpler fallback (to avoid pseudo-inverse cost), we compute the
    pairwise distance structure: sequences with similar k-mer profiles
    should have similar normalised embeddings.  Loss = MSE between the
    k-mer distance matrix and the embedding distance matrix (both
    normalised).

    Parameters
    ----------
    input_ids         : (B, L) token IDs — the raw DNA tokens.
    original_embeds   : (B, L, D) — embeddings before normalisation.
    normalized_embeds : (B, L, D) — embeddings after normalisation.
    vocab_size        : int — tokenizer vocabulary size.
    k                 : int — k-mer size for frequency counting.

    Returns
    -------
    Scalar loss >= 0.
    """
    B = input_ids.size(0)
    if B < 2:
        return torch.tensor(0.0, device=input_ids.device)

    # 1. Ground-truth k-mer frequency vectors (biological content)
    kmer_freq = _count_kmers(input_ids, vocab_size, k)  # (B, n_bins)

    # 2. Pairwise distance matrix from k-mer frequencies (biology)
    kmer_dist = torch.cdist(kmer_freq, kmer_freq, p=2)  # (B, B)

    # 3. Mean-pool normalised embeddings
    norm_mean = normalized_embeds.mean(dim=1)  # (B, D)

    # 4. Pairwise distance matrix from normalised embeddings
    emb_dist = torch.cdist(norm_mean, norm_mean, p=2)   # (B, B)

    # 5. Normalise both to [0, 1] for comparable scales
    kmer_max = kmer_dist.max().clamp(min=1e-8)
    emb_max = emb_dist.max().clamp(min=1e-8)
    kmer_norm = kmer_dist / kmer_max
    emb_norm = emb_dist / emb_max

    # 6. Loss: MSE between the two distance matrices
    #    If normalisation preserves biology, sequences with similar k-mer
    #    profiles will remain close in embedding space.
    loss = F.mse_loss(emb_norm, kmer_norm.detach())
    return loss
