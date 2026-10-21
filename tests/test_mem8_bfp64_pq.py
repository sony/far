"""Unit tests for the 8b-is Sovereign Omni BFP-64 + MEM|8 Adaptive Episodic
Memory Bank (`far.memories.mem8_bfp64_pq`). Runs on bare Python 3 stdlib or pytest.
"""
import math
import random
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "far" / "memories"))

from mem8_bfp64_pq import (  # noqa: E402
    Mem8AdaptiveEpisodicBankPq,
    Mem8EpisodicKeyRow,
)


class FarMem8Bfp64PqTest(unittest.TestCase):
    def test_bfp64_compression_is_3_94x_on_corridor_frames(self) -> None:
        bank = Mem8AdaptiveEpisodicBankPq(key_dim=64, chunk_size=3, num_sets=2, surprise_threshold=10.0)
        rng = random.Random(350)
        for t in range(30):
            vec = [rng.uniform(-0.5, 0.5) for _ in range(64)]
            bank.append_frame_key(vec, frame_time=float(t), interacted=False)

        self.assertEqual(bank.promoted_count, 0)
        self.assertAlmostEqual(bank.compression_ratio_vs_fp32, 256.0 / 65.0, places=5)

    def test_scale_invariant_integer_cosine_matches_float_cosine(self) -> None:
        rng = random.Random(42)
        bank = Mem8AdaptiveEpisodicBankPq(key_dim=64, chunk_size=3, num_sets=2, surprise_threshold=0.5)
        raw_keys = []
        for t in range(18):
            # Vary scale across frames by 100x to verify exponent cancellation in cosine
            scale = 0.05 if t % 2 == 0 else 5.0
            vec = [rng.uniform(-1.0, 1.0) * scale for _ in range(64)]
            raw_keys.append(vec)
            bank.append_frame_key(vec, frame_time=float(t), interacted=(t == 7))

        query = [rng.uniform(-1.0, 1.0) * 2.5 for _ in range(64)]
        cos_pq = bank.score_query_cosines_pq(query)

        q_norm = math.sqrt(sum(q * q for q in query))
        for t, (k_vec, c_pq) in enumerate(zip(raw_keys, cos_pq)):
            k_norm = math.sqrt(sum(k * k for k in k_vec))
            c_ref = sum(q * k for q, k in zip(query, k_vec)) / (q_norm * k_norm)
            tol = 2e-4 if bank.keys[t].promoted_i16 else 8e-3
            self.assertAlmostEqual(c_pq, c_ref, delta=tol)

    def test_ai2thor_drawer_revisit_salience_promotion_and_chunk_recall(self) -> None:
        """Simulates AI2-THOR Clip 350: agent explores a kitchen, puts a tomato
        in a drawer at frame 7 (chunk 2, middle frame), wanders through 20
        corridor frames, then revisits the drawer. Verifies MEM|8 promotes the
        interaction frame to BFP-i16 and recalls chunk 2 in Set 0, Rank 0.
        """
        rng = random.Random(350)
        bank = Mem8AdaptiveEpisodicBankPq(key_dim=64, chunk_size=3, num_sets=2, surprise_threshold=0.45)

        corridor_base = [rng.uniform(-0.3, 0.3) for _ in range(64)]
        drawer_tomato_key = [rng.uniform(-1.0, 1.0) for _ in range(64)]

        for t in range(30):
            if t == 7:
                # Frame 7 (inside chunk 2 = frames 6, 7, 8): tomato placed in drawer!
                bank.append_frame_key(drawer_tomato_key, frame_time=float(t), interacted=True)
            else:
                noise = [rng.uniform(-0.02, 0.02) for _ in range(64)]
                vec = [b + n for b, n in zip(corridor_base, noise)]
                bank.append_frame_key(vec, frame_time=float(t), interacted=False)

        self.assertTrue(bank.keys[7].promoted_i16)
        # Revisit query near drawer_tomato_key
        revisit_query = [v + rng.uniform(-0.05, 0.05) for v in drawer_tomato_key]
        recall = bank.recall_chunks_pq(revisit_query, num_memories=2, temperature=0.1)

        # Chunk 2 (containing frame 7) must be recalled first in Set 0!
        self.assertEqual(recall.selected_chunk_indices[0][0], 2)
        # Within chunk 2 (frames 6, 7, 8), middle frame index 1 (frame 7) must dominate the prior!
        self.assertGreater(recall.within_chunk_priors[0][0][1], 0.90)


if __name__ == "__main__":
    unittest.main()
