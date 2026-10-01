from __future__ import annotations

import random


class RandomClipSampler:
    def __init__(self, clip_len: int, frame_stride: int = 1) -> None:
        self.clip_len = clip_len
        self.frame_stride = frame_stride

    def __call__(self, sample: dict) -> dict:
        num_frames = int(sample["num_frames"])
        needed = 1 + (self.clip_len - 1) * self.frame_stride

        if num_frames <= needed:
            start_frame = 0
        else:
            max_start = num_frames - needed
            start_frame = random.randint(0, max_start)

        frame_indices = [min(start_frame + i * self.frame_stride, num_frames - 1) for i in range(self.clip_len)]
        return {
            "context_indices": frame_indices,
            "context_mask": [1] * self.clip_len,
            "goal_indices": [],
            "frame_indices": frame_indices,
        }


def _past_end(sample: dict, curr_frame_idx: int, pool_phase: str) -> int:
    """Exclusive upper bound of the frames the context pool may draw from.

    "past":     every frame before the current one (the original rule);
    "pre_eval": only frames before the backend's `curr_frame_idx` — the start
                of the evaluated phase (THOR: `eval_start`; LoopNav: the first
                return-leg frame; SoundSpaces: the first eval leg) — i.e. the
                pool is strictly the EXPLORATION history, never the agent's own
                eval-leg frames, matching the exploration-only rollout protocol.
                A current frame that itself precedes that point (event-anchored
                draws on outbound interactions) keeps its full past.
    """
    if pool_phase == "pre_eval":
        es = sample.get("curr_frame_idx")
        if es is not None and int(es) >= 1:
            return min(curr_frame_idx, int(es))
    return curr_frame_idx


def _build_context(pool_size: int, curr_frame_idx: int, past_end: int) -> tuple[list[int], list[int]]:
    """Random subsample of `range(past_end)` + the current frame at the tail;
    left-padded with index 0 / mask 0 when fewer than pool_size-1 past frames
    exist. `past_end == curr_frame_idx` reproduces the original draw exactly
    (same RNG consumption)."""
    if pool_size > past_end + 1:
        valid_indices = list(range(0, past_end)) + [curr_frame_idx]
        pad_count = pool_size - len(valid_indices)
        return [0] * pad_count + valid_indices, [0] * pad_count + [1] * len(valid_indices)
    sampled = sorted(random.sample(range(0, past_end), pool_size - 1))
    return sampled + [curr_frame_idx], [1] * pool_size


def eligible_event_spans(spans, num_frames: int, curr_frame_idx,
                         event_codes, event_phase: str, event_pre, event_post,
                         event_goal_anchor: str) -> list[tuple[int, int, int]]:
    """The interaction spans of one clip that `ContextGoalSampler._event_draw`
    can anchor a (current, goals) pair on — factored out so offline tooling
    (the event-index builder) applies EXACTLY the draw's eligibility and the
    two can never drift apart. `spans` is the `(code, start, end)` list from
    `far.data.datasets._event_spans`; empty/None input -> [].

    A span is eligible when its code passes `event_codes`, a current frame
    exists in `[start - pre_hi, start - pre_lo]` (clipped to the phase's first
    legal frame and, for negative pre, to the span's last frame), and a goal
    frame exists at `anchor + post_lo` or later inside the clip."""
    if not spans:
        return []
    lo_curr = int(curr_frame_idx) if (event_phase == "eval" and curr_frame_idx is not None) else 1
    pre_lo, pre_hi = event_pre
    post_lo = event_post[0]
    anchor_of = (lambda a, b: a) if event_goal_anchor == "start" else (lambda a, b: b)
    return [(c, a, b) for c, a, b in spans
            if (event_codes is None or c in event_codes)
            and min(pre_hi, a - lo_curr, a - 1) >= max(pre_lo, -(b - a))     # a current frame exists
            and anchor_of(a, b) + post_lo <= num_frames - 1]                 # a goal frame exists


class ContextGoalSampler:
    """
    Samples context_pool_size context frames ending at curr_frame_idx and goals_per_obs
    goal frames drawn uniformly without replacement from the future window
    (curr_frame_idx, curr_frame_idx + len_traj_pred] (capped by num_frames).

    curr_frame_idx is drawn uniformly from [len_traj_pred, num_frames - len_traj_pred - 1]
    to guarantee the full window is within bounds.

    EVENT BIAS (`event_bias_p` > 0; AI2-THOR). Interactions are 2-3-frame
    animations in ~900-frame episodes, so under the uniform draw a (current,
    goal) pair straddles one in well under 1% of samples — and when it does the
    goal sits anywhere in the next 6.4 s, so the `[open]` label is paired with
    mid-swing doors, fully-open doors and views where the agent has already
    walked off; the model learns their average (blurry half-open renders).
    With probability `event_bias_p` the pair is instead placed TIGHTLY around
    one interaction span `(code, start, end)` supplied by the dataset as
    `sample["event_spans"]` (derived from the backend's cumulative interaction
    counts): current = start - U[pre], every goal = anchor + U[post] with the
    anchor at the span START by default — so goals land DURING the 2-6-frame
    animation and just after it (the door mid-swing through just-opened), the
    frames where the label -> visible state change is sharpest. The context
    pool is built exactly as in the uniform branch.
    Falls back to the uniform draw when the episode has no eligible span (so
    corpora without events, and the p=0 default, are bit-identical to before).

    Args:
        event_bias_p:   probability of the event-anchored draw (0 = off).
        event_codes:    which interaction columns qualify (0 open / 1 close /
                        2 pickup / 3 put, the backend's INTERACT_CODES order);
                        None = all four (pickup/put are single-frame visual
                        changes too under the hidden-inventory carry rule).
        event_phase:    "any" — events anywhere in the episode (current may then
                        precede curr_frame_idx = eval_start; the pool is still
                        strictly the past); "eval" — only events at or after
                        curr_frame_idx, keeping the uniform branch's lower bound.
        event_pre:      (lo, hi) frames BEFORE the span start for the current
                        frame. lo may be <= 0: a negative offset puts the current
                        frame INSIDE the animation (e.g. -3 = 3 frames into the
                        swing, clipped to the span), the case a stride-2
                        rollout hits as soon as it has generated a half-open
                        door — the label then covers only the REMAINING fires
                        after the current frame, exactly as rollout computes it.
        event_post:     (lo, hi) goal offsets from the anchor (capped so
                        goal - current <= len_traj_pred, the trained horizon).
        event_goal_anchor: "start" (default) — offsets count from the span's
                        first animation frame, so post=(0, 8) covers the swing
                        and the first frames of the settled state; "end" —
                        offsets count from the last animation frame (goals only
                        in the settled state / dwell).
        pool_phase:     "past" — the pool draws from every frame before the
                        current one; "pre_eval" — strictly from frames before
                        the backend's `curr_frame_idx` (exploration history
                        only; see `_past_end`).
    """

    def __init__(
        self,
        context_pool_size: int,
        len_traj_pred: int,
        goals_per_obs: int = 1,
        event_bias_p: float = 0.0,
        event_codes: list[int] | None = None,
        event_phase: str = "any",
        event_pre: tuple[int, int] = (-3, 5),
        event_post: tuple[int, int] = (0, 8),
        event_goal_anchor: str = "start",
        event_len_traj_pred: int | None = None,
        pool_phase: str = "past",
        long_horizon_p: float = 0.0,
        long_horizon_min_past: int = 50,
    ) -> None:
        assert event_phase in ("any", "eval"), event_phase
        assert event_goal_anchor in ("start", "end"), event_goal_anchor
        assert pool_phase in ("past", "pre_eval"), pool_phase
        self.event_goal_anchor = event_goal_anchor
        # Horizon cap of the EVENT-ANCHORED draw only (frames; None -> len_traj_pred).
        # Lets the anchored pairs reach far (door-corridor: a pre-reveal current
        # to the SECOND end's reveal is up to 86 frames) while the uniform draw
        # keeps the short rollout-like horizon of len_traj_pred.
        self.event_len_traj_pred = int(event_len_traj_pred) if event_len_traj_pred else None
        self.pool_phase = pool_phase
        self.context_pool_size = context_pool_size
        self.len_traj_pred = len_traj_pred
        self.goals_per_obs = goals_per_obs
        self.event_bias_p = float(event_bias_p)
        # LONG-HORIZON pairs (retriever pre-training, `train_latent_retriever_hard.py`
        # mechanism 1): with probability long_horizon_p the current frame is drawn
        # from the PRE-eval span [long_horizon_min_past, curr_frame_idx) and the goals
        # from the eval leg [curr_frame_idx, num_frames) — "doorway view -> reveal".
        # 0 = off (bit-identical draw).
        assert 0.0 <= long_horizon_p <= 1.0, long_horizon_p
        self.long_horizon_p = float(long_horizon_p)
        self.long_horizon_min_past = int(long_horizon_min_past)
        self.event_codes = None if event_codes is None else {int(c) for c in event_codes}
        self.event_phase = event_phase
        self.event_pre = (int(event_pre[0]), int(event_pre[1]))
        self.event_post = (int(event_post[0]), int(event_post[1]))

    def _event_draw(self, sample: dict, num_frames: int):
        """(curr, goal_indices) tightly around one eligible interaction span, or None."""
        eligible = eligible_event_spans(
            sample.get("event_spans"), num_frames, sample.get("curr_frame_idx"),
            self.event_codes, self.event_phase, self.event_pre, self.event_post,
            self.event_goal_anchor)
        if not eligible:
            return None
        lo_curr = int(sample["curr_frame_idx"]) if (
            self.event_phase == "eval" and sample.get("curr_frame_idx") is not None) else 1
        pre_lo, pre_hi = self.event_pre
        post_lo, post_hi = self.event_post
        anchor_of = (lambda a, b: a) if self.event_goal_anchor == "start" else (lambda a, b: b)
        _, a, b = random.choice(eligible)
        # pre < 0 walks INTO the animation, never past its last frame.
        lo_pre = max(pre_lo, -(b - a))
        hi_pre = min(pre_hi, a - lo_curr, a - 1)                # current >= max(lo_curr, 1)
        if hi_pre < lo_pre:
            return None
        curr = a - random.randint(lo_pre, hi_pre)
        base = anchor_of(a, b)
        g_lo = max(base + post_lo, curr + 1)                    # goals strictly after current
        g_hi = min(base + post_hi, num_frames - 1, curr + (self.event_len_traj_pred or self.len_traj_pred))
        if g_hi < g_lo:
            return None
        goal_pool = list(range(g_lo, g_hi + 1))
        k = self.goals_per_obs
        goals = sorted(random.sample(goal_pool, k) if len(goal_pool) >= k else random.choices(goal_pool, k=k))
        return curr, goals

    def __call__(self, sample: dict) -> dict:
        num_frames = int(sample["num_frames"])
        hi = num_frames - 2

        # Event-anchored draw first. `force_event` (set by VideoDataset's
        # event-index substitution) overrides the Bernoulli: True -> the event
        # draw MUST succeed (the clip came from the event index, which was
        # built with this sampler's own eligibility predicate — a failure
        # means a stale index), False -> uniform draw, absent/None -> legacy
        # per-clip Bernoulli(event_bias_p).
        force_event = sample.get("force_event")
        if force_event:
            drawn = self._event_draw(sample, num_frames)
            if drawn is None:
                raise RuntimeError(
                    f"force_event: no feasible event draw on {sample.get('video_id')} — "
                    f"the event index is stale for these event params; rebuild it")
        elif force_event is None and self.event_bias_p > 0 and sample.get("event_spans") \
                and random.random() < self.event_bias_p:
            drawn = self._event_draw(sample, num_frames)
        else:
            drawn = None
        if drawn is not None:
            curr_frame_idx, goal_indices = drawn
            context_indices, context_mask = _build_context(
                self.context_pool_size, curr_frame_idx,
                _past_end(sample, curr_frame_idx, self.pool_phase))
            return {
                "context_indices": context_indices,
                "context_mask": context_mask,
                "goal_indices": goal_indices,
                "frame_indices": context_indices + goal_indices,
                "event_anchored": True,
            }

        # Long-horizon draw: current BEFORE the eval boundary, goals in the eval
        # leg (never on a forced event draw; needs the backend's eval boundary).
        es = sample.get("curr_frame_idx")
        if (self.long_horizon_p > 0 and es is not None and not force_event
                and random.random() < self.long_horizon_p):
            es = int(es)
            lo_c = min(self.long_horizon_min_past, max(es - 1, 0))
            if es - 1 >= lo_c and es <= num_frames - 1:
                curr_frame_idx = random.randint(lo_c, es - 1)
                goal_pool = list(range(es, num_frames))
                k = self.goals_per_obs
                goal_indices = sorted(random.sample(goal_pool, k) if len(goal_pool) >= k
                                      else random.choices(goal_pool, k=k))
                context_indices, context_mask = _build_context(
                    self.context_pool_size, curr_frame_idx,
                    _past_end(sample, curr_frame_idx, self.pool_phase))
                return {
                    "context_indices": context_indices,
                    "context_mask": context_mask,
                    "goal_indices": goal_indices,
                    "frame_indices": context_indices + goal_indices,
                    "event_anchored": False,
                    "long_horizon": True,
                }

        if sample.get("curr_frame_idx") is not None:
            lo = sample["curr_frame_idx"]
            curr_frame_idx = random.randint(lo, hi) if lo < hi else lo
        else:
            curr_frame_idx = random.randint(0, hi) if hi > 0 else 0

        context_indices, context_mask = _build_context(
            self.context_pool_size, curr_frame_idx,
            _past_end(sample, curr_frame_idx, self.pool_phase))

        goal_pool = list(range(curr_frame_idx + 1, min(curr_frame_idx + self.len_traj_pred + 1, num_frames)))

        k = self.goals_per_obs
        goal_indices = sorted(random.sample(goal_pool, k) if len(goal_pool) >= k else random.choices(goal_pool, k=k))

        return {
            "context_indices": context_indices,
            "context_mask": context_mask,
            "goal_indices": goal_indices,
            "frame_indices": context_indices + goal_indices,
            "event_anchored": False,
        }


class ContextTrajectorySampler:
    """
    Samples a context pool ending at curr_frame_idx (same selection rules as
    ContextGoalSampler: random subsample of the past + the current frame,
    left-padded with index 0 / mask 0 when the clip is short) plus the
    contiguous len_traj_pred frames immediately following curr_frame_idx as
    goals.

    curr_frame_idx is taken DIRECTLY from `sample["curr_frame_idx"]` when the
    backend provides one (LoopNav: the start of the return leg — so every
    trajectory starts at the loop-closure point, deterministically); only
    backends without the concept (curr_frame_idx=None) fall back to a uniform
    draw over [0, num_frames - len_traj_pred - 1]. Goal indices past the clip
    end repeat the final frame, so goal count is ALWAYS len_traj_pred and
    batches collate to fixed shapes.
    """

    def __init__(self, context_pool_size: int, len_traj_pred: int,
                 pool_phase: str = "past") -> None:
        assert pool_phase in ("past", "pre_eval"), pool_phase
        self.context_pool_size = context_pool_size
        self.len_traj_pred = len_traj_pred
        self.pool_phase = pool_phase          # see `_past_end` (same knob as ContextGoalSampler)

    def __call__(self, sample: dict) -> dict:
        num_frames = int(sample["num_frames"])

        if sample.get("curr_frame_idx") is not None:
            curr_frame_idx = int(sample["curr_frame_idx"])
        else:
            hi = max(num_frames - self.len_traj_pred - 1, 0)
            curr_frame_idx = random.randint(0, hi) if hi > 0 else 0

        context_indices, context_mask = _build_context(
            self.context_pool_size, curr_frame_idx,
            _past_end(sample, curr_frame_idx, self.pool_phase))

        # Contiguous horizon, ALWAYS len_traj_pred long: indices past the clip
        # end repeat the final frame (same repeat-padding convention as the
        # context pool) so batched collate sees fixed shapes.
        goal_indices = [min(curr_frame_idx + 1 + i, num_frames - 1) for i in range(self.len_traj_pred)]
        return {
            "context_indices": context_indices,
            "context_mask": context_mask,
            "goal_indices": goal_indices,
            "frame_indices": context_indices + goal_indices,
        }
