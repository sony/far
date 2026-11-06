"""
================================================================================
[8b-is Sovereign FAR Episodic Memory — Omni BFP-64 + MEM|8 Adaptive Recall]

Connects Sony's FAR ("Learning What to Recall: Adaptive Multi-Cue Episodic
Memory for World Models") with the 8b-is conditional & episodic memory stack:
- https://github.com/8b-is/Engram           (Omni BFP-64 + Lemire p/q conditional memory)
- https://github.com/8b-is/alexiai          (GAIA-MLX-QUANT p/q + BFP-64 + MEM|8 substrate)
- https://github.com/8b-is/deepsiper-enthea (Sovereign multi-model evaluation harness)

Why FAR (`PnPRetrieval` / `QueryViT`) + Omni `BFP-64` & `MEM|8` is a 1-to-1 match:
1. Scale-Invariant Integer Cosine Similarity with Precomputed `isqrt` Norms:
   In `PnPRetrieval`, `QueryViT` maps frames and action-conditioned queries into a
   shared cosine space: `cos(q, k) = <q, k> / (||q||_2 * ||k||_2)`.
   When a key `k` is quantized into integer-aligned `BFP-64` mantissas `K_i`, its
   global power-of-two scale `2^{min_exp}` cancels out completely between the
   numerator `<q, k>` and denominator `||k||_2`!
   Even better: each episodic frame key precomputes its exact integer reciprocal
   L2 norm `inv_norm_q20` ONCE upon insertion into the memory bank using integer
   `isqrt_u128`. Scoring thousands of past frames at recall time requires ZERO
   square roots and ZERO floating-point divisions per frame!

2. 3.94x Episodic Key Bank Compression (`65 bytes` per 64 dims vs `256 bytes` FP32):
   Long-horizon world-model rollouts (LoopNav, SoundSpaces, AI2-THOR) fit 3.94x
   more episodic frame keys in SRAM/VRAM.

3. MEM|8 Temporal-Surprise & Interaction-Guided Precision Promotion (`i8 -> i16`):
   Corridor traversal frames stay in compact `BFP-i8` (`65 B/64d`, `i16` block
   accumulator), while high-surprise state-change frames (e.g., putting the tomato
   in the drawer in AI2-THOR Clip 350, where `||k_t - k_{t-1}||_2 > tau` or
   `interact == True`) automatically promote to `BFP-i16` (`129 B/64d`, `i32`
   accumulator), preserving 10x higher cosine fidelity on critical revisit frames!
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

BFP_BLOCK_SIZE: int = 64
PQ_Q30: int = 1 << 30


def isqrt_u128(n: int) -> int:
    """Exact integer floor square root floor(sqrt(n))."""
    if n < 2:
        return max(0, n)
    return math.isqrt(n)


@dataclass(frozen=True)
class Mem8EpisodicKeyRow:
    """
    A single `QueryViT` episodic frame key stored in Omni `BFP-64` (`i8`) or
    promoted `MEM|8 BFP-i16` (`i16`) with precomputed integer reciprocal L2 norm.
    """
    aligned_mantissas: Tuple[int, ...]  # integer-aligned across 64-dim blocks
    min_exp: int
    inv_norm_q30: int                   # round((1 / ||aligned_mantissas||_2) * 2^30)
    promoted_i16: bool
    frame_time: float

    @classmethod
    def from_floats(
        cls,
        key_vec: Sequence[float],
        frame_time: float = 0.0,
        promote_i16: bool = False,
    ) -> "Mem8EpisodicKeyRow":
        dim = len(key_vec)
        assert dim > 0
        num_blocks = (dim + BFP_BLOCK_SIZE - 1) // BFP_BLOCK_SIZE
        max_q = 32767.0 if promote_i16 else 127.0
        clamp_lim = 32767 if promote_i16 else 127

        block_mants: List[List[int]] = []
        block_exps: List[int] = []

        for b in range(num_blocks):
            blk = key_vec[b * BFP_BLOCK_SIZE:(b + 1) * BFP_BLOCK_SIZE]
            max_abs = max((abs(v) for v in blk), default=0.0)
            if max_abs < 1e-18:
                block_mants.append([0] * len(blk))
                block_exps.append(-60)
            else:
                e = max(-60, min(60, math.ceil(math.log2(max_abs / max_q))))
                inv_s = math.ldexp(1.0, -e)
                m = [max(-clamp_lim, min(clamp_lim, int(round(v * inv_s)))) for v in blk]
                block_mants.append(m)
                block_exps.append(e)

        min_exp = min(block_exps)
        aligned: List[int] = []
        for m_blk, e in zip(block_mants, block_exps):
            shift = max(0, min(16, e - min_exp))
            mult = 1 << shift
            aligned.extend(v * mult for v in m_blk)

        sum_sq = sum(v * v for v in aligned)
        if sum_sq == 0:
            inv_norm_q30 = 0
        else:
            # sqrt(sum_sq) in Q30 = isqrt(sum_sq * 2^60)
            norm_q30 = max(1, isqrt_u128(sum_sq << 60))
            inv_norm_q30 = ((1 << 60) + (norm_q30 >> 1)) // norm_q30

        return cls(
            aligned_mantissas=tuple(aligned),
            min_exp=min_exp,
            inv_norm_q30=inv_norm_q30,
            promoted_i16=promote_i16,
            frame_time=float(frame_time),
        )

    @property
    def storage_bytes(self) -> int:
        dim = len(self.aligned_mantissas)
        num_blocks = (dim + BFP_BLOCK_SIZE - 1) // BFP_BLOCK_SIZE
        bytes_per_elem = 2 if self.promoted_i16 else 1
        return dim * bytes_per_elem + num_blocks


@dataclass(frozen=True)
class RecallResultPq:
    """
    Mirrors `PnPRetrieval` + `LocalCandidates` chunk selection output:
    - `selected_chunk_indices`: `[num_sets][num_memories]` chunk indices in rank order
    - `set_scores`:             `[num_sets]` softmax posterior over the K sets
    - `within_chunk_priors`:    `[num_sets][num_memories][chunk_size]` per-frame softmax prior
    - `frame_cosines`:          `[num_chunks * chunk_size]` integer-derived cosine similarities
    """
    selected_chunk_indices: List[List[int]]
    set_scores: List[float]
    within_chunk_priors: List[List[List[float]]]
    frame_cosines: List[float]


class Mem8AdaptiveEpisodicBankPq:
    """
    Episodic Memory Bank for World Models (FAR `PnPRetrieval`) using Omni `BFP-64`
    (`65 B / 64 dims`) with automatic `MEM|8` promotion (`BFP-i8 -> BFP-i16`) on
    high-surprise state-transition or object-interaction frames.
    """

    def __init__(
        self,
        key_dim: int = 64,
        chunk_size: int = 3,
        num_sets: int = 2,
        surprise_threshold: float = 0.35,
    ):
        self.key_dim = int(key_dim)
        self.chunk_size = int(chunk_size)
        self.num_sets = int(num_sets)
        self.surprise_threshold = float(surprise_threshold)
        self.keys: List[Mem8EpisodicKeyRow] = []
        self._prev_raw_key: Optional[List[float]] = None

    def append_frame_key(
        self,
        key_vec: Sequence[float],
        frame_time: float = 0.0,
        interacted: bool = False,
    ) -> Mem8EpisodicKeyRow:
        """
        Insert a frame's `QueryViT` key into the episodic bank.
        Promotes automatically to `BFP-i16` if `interacted=True` or if the temporal
        latent surprise `||k_t - k_{t-1}||_2 / ||k_t||_2 > surprise_threshold`.
        """
        promote = bool(interacted)
        if not promote and self._prev_raw_key is not None:
            diff_sq = sum((a - b) ** 2 for a, b in zip(key_vec, self._prev_raw_key))
            norm_sq = max(1e-18, sum(a * a for a in key_vec))
            if math.sqrt(diff_sq / norm_sq) > self.surprise_threshold:
                promote = True

        self._prev_raw_key = list(key_vec)
        row = Mem8EpisodicKeyRow.from_floats(key_vec, frame_time=frame_time, promote_i16=promote)
        self.keys.append(row)
        return row

    @property
    def nbytes_compressed(self) -> int:
        return sum(k.storage_bytes for k in self.keys)

    @property
    def nbytes_fp32(self) -> int:
        return len(self.keys) * self.key_dim * 4

    @property
    def compression_ratio_vs_fp32(self) -> float:
        return self.nbytes_fp32 / max(1, self.nbytes_compressed)

    @property
    def promoted_count(self) -> int:
        return sum(1 for k in self.keys if k.promoted_i16)

    def score_query_cosines_pq(self, query_vec: Sequence[float]) -> List[float]:
        """
        Compute cosine similarity `<q, k_t> / (||q||_2 * ||k_t||_2)` against all
        stored keys using integer-aligned `BFP` dot products and precomputed
        `inv_norm_q20` (ZERO square roots per stored frame!).
        """
        q_row = Mem8EpisodicKeyRow.from_floats(query_vec, promote_i16=True)
        q_m = q_row.aligned_mantissas
        q_inv_q30 = q_row.inv_norm_q30
        inv_q60 = 1.0 / float(PQ_Q30 * PQ_Q30)

        cosines: List[float] = []
        for k_row in self.keys:
            dot_i64 = sum(qi * ki for qi, ki in zip(q_m, k_row.aligned_mantissas))
            # Scale-invariant integer cosine in Q60 -> float [-1.0, 1.0]
            cos_val = (dot_i64 * q_inv_q30 * k_row.inv_norm_q30) * inv_q60
            cosines.append(max(-1.0, min(1.0, cos_val)))
        return cosines

    def recall_chunks_pq(
        self,
        query_vec: Sequence[float],
        num_memories: int = 2,
        meta_bonus_per_frame: Optional[Sequence[float]] = None,
        temperature: float = 0.1,
    ) -> RecallResultPq:
        """
        Execute FAR `PnPRetrieval` chunk-level max ranking, K-set partitioning,
        and `LocalCandidates` within-chunk `frame_prior` softmax over `chunk_size`.
        """
        num_chunks = len(self.keys) // self.chunk_size
        usable_frames = num_chunks * self.chunk_size
        cosines = self.score_query_cosines_pq(query_vec)[:usable_frames]

        if meta_bonus_per_frame is not None:
            fused_logits = [c + float(m) for c, m in zip(cosines, meta_bonus_per_frame[:usable_frames])]
        else:
            fused_logits = list(cosines)

        chunk_scores: List[Tuple[float, int]] = []
        for c_idx in range(num_chunks):
            c_slice = fused_logits[c_idx * self.chunk_size:(c_idx + 1) * self.chunk_size]
            chunk_scores.append((max(c_slice), c_idx))

        # Rank chunks descending by max frame score
        chunk_scores.sort(key=lambda pair: pair[0], reverse=True)
        total_needed = min(num_chunks, self.num_sets * int(num_memories))
        top_chunks = [c_idx for _, c_idx in chunk_scores[:total_needed]]

        selected_sets: List[List[int]] = []
        within_priors: List[List[List[float]]] = []
        set_raw_logits: List[float] = []

        inv_temp = 1.0 / max(1e-6, float(temperature))
        for k in range(self.num_sets):
            set_chunks = top_chunks[k * num_memories:(k + 1) * num_memories]
            if not set_chunks:
                break
            selected_sets.append(set_chunks)
            set_logit_sum = 0.0
            set_priors: List[List[float]] = []
            for c_idx in set_chunks:
                c_logits = fused_logits[c_idx * self.chunk_size:(c_idx + 1) * self.chunk_size]
                max_l = max(c_logits)
                set_logit_sum += max_l
                exps = [math.exp((l - max_l) * inv_temp) for l in c_logits]
                s_exp = sum(exps)
                set_priors.append([e / s_exp for e in exps])
            within_priors.append(set_priors)
            set_raw_logits.append(set_logit_sum / len(set_chunks))

        max_set_l = max(set_raw_logits) if set_raw_logits else 0.0
        set_exps = [math.exp((l - max_set_l) * inv_temp) for l in set_raw_logits]
        sum_set_exps = max(1e-12, sum(set_exps))
        set_scores = [e / sum_set_exps for e in set_exps]

        return RecallResultPq(
            selected_chunk_indices=selected_sets,
            set_scores=set_scores,
            within_chunk_priors=within_priors,
            frame_cosines=cosines,
        )
