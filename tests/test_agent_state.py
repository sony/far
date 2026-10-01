"""Unit tests for the agent-state observation: the backend's per-frame
second-agent detection (`THORDynBackend(agent_state=True)`), the
`AgentStatePredictor`, the `AgentStateCDiT` composite generator, `LDWMCore`'s
agent-state protocol (loss, RAG rewards, sampling) and the retriever's
agent-state cue. Plain asserts — runs under pytest or directly:

    python tests/test_agent_state.py

Tests that need the dynamic-agent corpus / a checkpoint on disk skip themselves
when the files are absent (reported as skipped under pytest), so the
pure-tensor tests run anywhere.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
os.environ.setdefault("FAR_RESULT_ROOT", str(ROOT / "results"))
os.environ.setdefault("FAR_MODELS_ROOT", str(ROOT / "models"))

from far.generators.agent_state import AgentStatePredictor, AgentStateCDiT   # noqa: E402
from far.generators.cdit_mod_pose_fa import CDiT as PoseFaCDiT               # noqa: E402

DYN_EP = ROOT / "datasets/ai2thor_dyn/latent/T9_2_00175_act"
DYN_RUN = ROOT / "models/ai2thor_dyn/far_multicue_noagent"   # a pose_fa (pre-agent-state) dyn run; the test skips when absent
SMALL_DIT = dict(in_height=32, in_width=32, in_channels=4, context_size=4, patch_size=4, hidden_size=64,
                 depth=2, num_heads=4, action_dim=5, pose_dim=5, interact_dim=5, stride_to_seconds=1.0)


def _skip(msg):
    """Skip the calling test: under pytest via `pytest.skip` (reported as
    skipped, not passed); without pytest print the reason and return True."""
    try:
        import pytest
    except ImportError:
        print(f"  SKIP: {msg}")
        return True
    pytest.skip(msg)


# ── §3.1 data ────────────────────────────────────────────────────────────────
def test_backend_state_geometry():
    """`[vi, dx, dy]` is A2 in the frame's own egocentric floor frame: whenever
    A2 is on screen it must sit in FRONT of the camera and inside the
    horizontal FOV, and the primed half must be the frame `pair_s` earlier."""
    if not DYN_EP.is_dir():
        return _skip("dyn corpus not on disk")
    from far.data.backends import THORDynBackend
    be = THORDynBackend(agent_state=True, agent_state_pair_s=1.5, agent_state_px_min=1)
    out = be.load_clip({"path": str(DYN_EP)})
    assert len(out) == 10, len(out)
    frames, pose, state = out[0], out[1], out[9]
    T = frames.shape[0]
    assert state.shape == (T, 6) and state.dtype == torch.float32
    px = np.load(DYN_EP / "agent2.npz")["px"][:T]
    vi = state[:, 0].numpy()
    assert np.array_equal(vi, (px >= 1).astype(np.float32))
    assert torch.all(state[vi == 0][:, 1:3] == 0), "positions must be zero when not in view"
    fwd, lat = state[vi == 1, 1].numpy(), state[vi == 1, 2].numpy()
    bearing = np.degrees(np.arctan2(lat, fwd))
    frac_ok = np.mean((fwd > 0) & (np.abs(bearing) <= 50))
    assert frac_ok >= 0.95, f"only {frac_ok:.0%} of visible frames have A2 in front / in FOV: sign convention wrong?"
    assert np.median(np.hypot(fwd, lat)) > 0.5, "A2 range implausibly small"
    # SIGN CONVENTION against the world: the last crossing heads to the FINAL
    # end, so its lateral sweep must point toward that end as seen from the
    # watch pose (ends from meta, Unity (x, z) -> standard [z, x]); the crossing
    # before it heads the other way.
    meta = json.loads((DYN_EP / "meta.json").read_text())
    legs = {L["name"]: (L["start"], L["end"]) for L in meta["legs"]}
    a, b = legs["watch"]
    on = vi[a:b] > 0; e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0])); pulses = []
    for s0, s1 in zip(e[::2] + a, e[1::2] + a):
        if pulses and s0 - pulses[-1][1] < 20: pulses[-1] = (pulses[-1][0], s1)
        else: pulses.append((s0, s1))
    assert len(pulses) == meta["K"], (len(pulses), meta["K"])
    def lat_of(pt_xz, f):        # lateral coordinate of a world point in frame f's egocentric frame
        p = np.array([pt_xz[1], pt_xz[0]]); o = pose[f, :2].numpy(); yaw = float(pose[f, 4])
        d = p - o; return -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]
    fin, other = meta["final"], ("B" if meta["final"] == "A" else "A")
    for k, (s0, s1) in enumerate(reversed(pulses[-2:])):          # k=0 last crossing, k=1 the one before
        f0, f1 = s0, s1 - 1
        toward = lat_of(meta[f"end_{fin if k == 0 else other}"], f1) - lat_of(meta[f"end_{other if k == 0 else fin}"], f1)
        sweep = float(state[f1, 2] - state[f0, 2])
        assert np.sign(sweep) == np.sign(toward), f"crossing {k} sweeps {sweep:+.2f} m but its end lies at {toward:+.2f} m laterally"
    # pair lookup
    j = np.clip(np.arange(T) - 15, 0, T - 1)
    assert torch.equal(state[:, 3:], state[j, :3])
    # threshold and noise knobs
    be2 = THORDynBackend(agent_state=True, agent_state_px_min=500)
    s2 = be2.load_clip({"path": str(DYN_EP)})[9]
    assert s2[:, 0].sum() < state[:, 0].sum()
    be3 = THORDynBackend(agent_state=True, agent_state_noise=0.05)
    s3a, s3b = be3.load_clip({"path": str(DYN_EP)})[9], be3.load_clip({"path": str(DYN_EP)})[9]
    assert torch.equal(s3a, s3b), "noise must be deterministic per episode"
    assert not torch.equal(s3a[:, 1:3], state[:, 1:3])
    # off -> slot is None; the reveal spans still travel in slot 9
    assert THORDynBackend(agent_state=False).load_clip({"path": str(DYN_EP)})[9] is None
    print(f"  backend: T={T}, visible {int(vi.sum())} frames, {frac_ok:.0%} in front/in-FOV, "
          f"median range {np.median(np.hypot(fwd, lat)):.2f} m")


def test_dataset_and_formatter():
    """The real train pipeline (Hydra config, sampler, formatter) emits
    `context_agent_state (T, 6)` / `goal_agent_state (G, 3)` aligned with the
    context / goal frames; datasets without the knob collate unchanged."""
    if not DYN_EP.is_dir():
        return _skip("dyn corpus not on disk")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from far.constants import CONFIG_ROOT
    ovs = ["datasets=[ai2thor_dyn_latent]", "len_traj_pred=32",
           "ai2thor_dyn_latent.shared.backend.mask_suffix=.agent.npy",
           "ai2thor_dyn_latent.shared.backend.agent_state=true",
           "ai2thor_dyn_latent.train.base_dataset.event_frac=1.0"]      # every draw reveal-anchored
    with initialize_config_dir(version_base=None, config_dir=CONFIG_ROOT):
        cfg = compose(config_name="world_train", overrides=ovs)
    ds = instantiate(cfg.ai2thor_dyn_latent.train)
    base = getattr(ds, "base_dataset", None) or getattr(ds, "dataset")
    fmt = getattr(ds, "formatter")
    import random
    random.seed(0); torch.manual_seed(0)
    item = base[0]
    out = fmt(item)
    n, g = len(item["context_indices"]), len(item["goal_indices"])
    assert out["context_agent_state"].shape == (n, 6) and out["goal_agent_state"].shape == (g, 3)
    assert out["context_frames"].shape[0] == n and out["goal_frames"].shape[0] == g
    # alignment with the raw per-frame state (waypoint units = metres here)
    raw = item["agent_state"]
    mpw = float(cfg.ai2thor_dyn_latent.spec.meters_per_waypoint)
    exp_ctx = raw[:n].clone(); exp_ctx[:, 1:3] /= mpw; exp_ctx[:, 4:6] /= mpw
    assert torch.allclose(out["context_agent_state"], exp_ctx)
    assert torch.allclose(out["goal_agent_state"][:, 0], raw[n:n + g, 0])
    # anchored draw: the current frame is agent-free, and the goal's presence label
    # is 1 iff the goal frame shows A2 (agent2 px > 0 is the visibility authority)
    assert out["context_agent_state"][-1, 0] == 0, "anchored current must be agent-free"
    gi = torch.as_tensor(item["goal_indices"])
    # vi at the goal must agree with the goal-row mask presence (same 'in view' notion)
    if "goal_obj" in out:
        agree = ((out["goal_agent_state"][:, 0] > 0) == (out["goal_obj"] > 0)).float().mean()
        assert agree >= 0.75, f"goal vi vs silhouette presence agree only {agree:.0%}"
    # knob off -> keys absent, everything else identical
    with initialize_config_dir(version_base=None, config_dir=CONFIG_ROOT):
        cfg0 = compose(config_name="world_train", overrides=ovs[:-2] + [ovs[-1]])
    ds0 = instantiate(cfg0.ai2thor_dyn_latent.train)
    random.seed(0); torch.manual_seed(0)
    out0 = ds0[0]
    assert "context_agent_state" not in out0 and "goal_agent_state" not in out0
    assert torch.equal(out0["context_frames"], out["context_frames"])
    print(f"  dataset: n={n} g={g}, goal vi={out['goal_agent_state'][:, 0].tolist()}, "
          f"context agent frames {int(out['context_agent_state'][:, 0].sum())}/{n}")


# ── §3.2 predictor ───────────────────────────────────────────────────────────
def _inputs(N, T, seed=0, mask=None):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)
    state = r(N, T, 6); state[..., 0] = (state[..., 0] > 0).float(); state[..., 3] = (state[..., 3] > 0).float()
    return dict(slot_state=state, slot_pose=r(N, T, 5), slot_time=torch.rand(N, T, generator=gen) * 40,
                slot_mask=mask, action=r(N, 5), stride=torch.rand(N, generator=gen) * 3, cur_time=torch.full((N,), 39.0))


def test_predictor_shapes_and_invariances():
    torch.manual_seed(0)
    p = AgentStatePredictor(dim=32, heads=4).eval()
    for T in (1, 3, 11):
        out = p(**_inputs(4, T))
        assert out.shape == (4, 3) and torch.isfinite(out).all(), T
    # permutation invariance over slots (with a consistent mask)
    N, T = 3, 6
    kw = _inputs(N, T, seed=1, mask=torch.tensor([[1, 1, 1, 1, 0, 0]] * N, dtype=torch.bool))
    ref = p(**kw)
    perm = torch.randperm(T)
    kw2 = {k: (v[:, perm] if k in ("slot_state", "slot_pose", "slot_time", "slot_mask") else v) for k, v in kw.items()}
    assert torch.allclose(p(**kw2), ref, atol=1e-5), "not permutation-invariant"
    # padded slots must not influence the output
    kw3 = _inputs(N, T, seed=1, mask=kw["slot_mask"])
    kw3["slot_state"][:, 4:] = 99.0; kw3["slot_pose"][:, 4:] = -99.0
    assert torch.allclose(p(**kw3), ref, atol=1e-5), "padding leaks"
    # every slot masked -> the null token alone, still finite
    kw4 = _inputs(N, T, seed=1, mask=torch.zeros(N, T, dtype=torch.bool))
    assert torch.isfinite(p(**kw4)).all()
    # slots DO matter when unmasked
    kw5 = _inputs(N, T, seed=2, mask=kw["slot_mask"])
    assert not torch.allclose(p(**kw5), ref, atol=1e-3)
    print("  predictor: shapes OK for T=1/3/11, permutation-invariant, padding-blind, null-only finite")


def test_predictor_velocity_feature():
    """`velocity_in_current_frame` rotates the slot's velocity by the slot's
    relative yaw: a slot looking 90 deg left that sees A2 moving forward
    reports A2 moving LEFT in the current frame; zero unless both paired
    observations are in view; relative times make the output invariant to a
    common shift of every absolute time."""
    torch.manual_seed(0)
    p = AgentStatePredictor(dim=32, heads=4).eval()
    kw = _inputs(1, 1)
    # slot: yaw +90 deg; A2 now at (2, 0) forward, 1.5 s ago at (1, 0) -> velocity +1 forward in the slot frame
    kw["slot_pose"] = torch.tensor([[[0.0, 0.0, 0.0, 0.0, torch.pi / 2]]])
    kw["slot_state"] = torch.tensor([[[1.0, 2.0, 0.0, 1.0, 1.0, 0.0]]])
    feats = {}
    h = p.slot_mlp.register_forward_pre_hook(lambda m, i: feats.__setitem__("in", i[0].detach().clone()))
    p(**kw)
    v = feats["in"][0, 0, 6:8]                                      # the two appended dims follow state6
    assert torch.allclose(v, torch.tensor([0.0, 1.0]), atol=1e-6), v
    kw["slot_state"][0, 0, 3] = 0.0                                 # partner not in view -> no velocity
    p(**kw); h.remove()
    assert torch.equal(feats["in"][0, 0, 6:8], torch.zeros(2)), feats["in"][0, 0, 6:8]
    # absolute-time invariance
    kw = _inputs(3, 4, seed=5); a = p(**kw)
    kw["slot_time"] = kw["slot_time"] + 123.0; kw["cur_time"] = kw["cur_time"] + 123.0
    kw["stride"] = kw["stride"] + 123.0 / p.stride_to_seconds
    assert torch.allclose(p(**kw), a, atol=1e-4)
    print("  velocity feature: rotation into the current frame OK, gated on both halves, time-shift invariant")


def test_predictor_loss():
    p = AgentStatePredictor(dim=32, heads=4, pos_loss_weight=2.0)
    pred = torch.tensor([[3.0, 1.0, 1.0], [-3.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
    tgt = torch.tensor([[1.0, 0.0, 0.0], [0.0, 9.0, 9.0], [1.0, 1.0, 0.0]])
    L = p.loss(pred, tgt)
    assert L.shape == (3,)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 0], tgt[:, 0], reduction="none")
    assert torch.allclose(L[1], bce[1]), "position term must vanish when the agent is absent"
    assert torch.allclose(L[0], bce[0] + 2.0 * 2.0)
    assert torch.allclose(L[2], bce[2] + 2.0 * 1.0)
    # gradients reach every parameter through forward + loss
    kw = _inputs(4, 5); out = p(**kw); p.loss(out, torch.rand(4, 3)).mean().backward()
    n_grad = sum(int(q.grad is not None and q.grad.abs().sum() > 0) for q in p.parameters())
    assert n_grad >= len(list(p.parameters())) - 1, n_grad     # (null token may get zero grad when never attended)
    print("  predictor loss: masking and gradients OK")


# ── §3.2 composite generator ────────────────────────────────────────────────
def _dit_inputs(N=2, T=4, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(x=r(N, C, 32, 32), t=torch.randint(0, 1000, (N,), generator=g), action=r(N, 5) * 0.1,
                context=r(N, T, C, 32, 32), stride=torch.rand(N, generator=g),
                context_time=torch.sort(torch.rand(N, T, generator=g) * 40, dim=1).values,
                context_pose=r(N, T, 5) * 0.1, action_interact=torch.zeros(N, 5))


def test_composite_identity_and_delegation():
    torch.manual_seed(0)
    comp = AgentStateCDiT(predictor_dim=32, predictor_heads=4, **SMALL_DIT).eval()
    # A FRESH DiT is adaLN-zero with a zero final projection: it outputs exactly
    # zero for any input, which would make every equality below vacuous. Give
    # the gates and the final layer weight so the forward is non-trivial (the
    # trained-checkpoint test below repeats the checks on real weights).
    with torch.no_grad():
        comp.dit.final_layer.linear.weight.normal_(std=0.02)
        for m in comp.dit.modules():
            if hasattr(m, "adaLN_modulation"):
                m.adaLN_modulation[-1].weight.normal_(std=0.02)
    bare = PoseFaCDiT(**SMALL_DIT).eval()
    bare.load_state_dict(comp.dit.state_dict(), strict=True)
    for a in ("context_size", "interact_dim", "in_channels", "out_channels", "patch_size"):
        assert getattr(comp, a) == getattr(bare, a), a
    kw = _dit_inputs()
    with torch.no_grad():
        ref = bare(**kw)
        no_state = comp(**kw)
        with_state = comp(**kw, agent_state=torch.randn(2, 3) * 5)
    assert ref.abs().max() > 0, "test DiT must produce a non-trivial output"
    assert torch.equal(no_state, ref), "composite without state must equal the bare DiT"
    assert torch.equal(with_state, ref), "zero-init conditioning must be a no-op at step 0"
    # once the conditioning has weight, the state changes the output
    with torch.no_grad():
        comp.state_cond[2].weight.normal_(std=0.1)
        assert not torch.allclose(comp(**kw, agent_state=torch.randn(2, 3) * 5), ref)
    # predict_agent / agent_loss protocol
    pk = _inputs(2, 4)
    pred = comp.predict_agent(**pk)
    assert pred.shape == (2, 3) and comp.agent_loss(pred, torch.rand(2, 3)).shape == (2,)
    # a FLAT pose_fa checkpoint warm-starts the DiT strictly; own checkpoints round-trip
    torch.save({"ema": bare.state_dict()}, "/tmp/_flat_pose_fa.pth.tar")
    comp2 = AgentStateCDiT(predictor_dim=32, predictor_heads=4, **SMALL_DIT).eval()
    comp2.load_weights("/tmp/_flat_pose_fa.pth.tar")
    with torch.no_grad():
        assert torch.equal(comp2(**kw), ref)
    torch.save({"ema": {"_orig_mod." + k: v for k, v in comp.state_dict().items()}}, "/tmp/_comp.pth.tar")
    comp3 = AgentStateCDiT(predictor_dim=32, predictor_heads=4, **SMALL_DIT).eval()
    comp3.load_weights("/tmp/_comp.pth.tar")
    assert all(torch.equal(a, b) for a, b in zip(comp3.state_dict().values(), comp.state_dict().values()))
    # the loss trains only the predictor; the DiT's conditioning path is detached at the core
    n_pred = sum(q.numel() for q in comp.predictor.parameters())
    print(f"  composite: identity at step 0 OK, delegation OK, flat warm start OK, predictor {n_pred / 1e3:.0f}k params")


def test_warm_start_real_checkpoint():
    """The composite built from a real dyn run's generator config loads that
    run's flat checkpoint strictly and reproduces its forward."""
    ck = DYN_RUN / "checkpoints/latest.pth.tar"
    if not ck.is_file():
        return _skip("dynamic-agent checkpoint not on disk")
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    cfg = OmegaConf.load(DYN_RUN / "config.yaml")
    gcfg = OmegaConf.to_container(cfg.model.generator, resolve=True); gcfg.pop("ckpt_path", None)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bare = instantiate(OmegaConf.create(gcfg), in_channels=4, in_height=32, in_width=32).eval().to(dev)
    sd = {k.removeprefix("_orig_mod."): v for k, v in torch.load(ck, map_location="cpu", weights_only=False)["ema"].items()}
    bare.load_state_dict(sd, strict=True)
    gcfg["_target_"] = "far.generators.agent_state.AgentStateCDiT"
    comp = instantiate(OmegaConf.create(gcfg), in_channels=4, in_height=32, in_width=32).eval().to(dev)
    comp.load_weights(str(ck))
    kw = {k: v.to(dev) for k, v in _dit_inputs(N=1, T=int(bare.context_size), seed=3).items()}
    st = torch.tensor([[1.0, 2.0, -0.5]], device=dev)
    with torch.no_grad():
        ref, out = bare(**kw), comp(**kw, agent_state=st)
        assert torch.equal(out, ref)
        # on TRAINED adaLN gates the conditioning path is live as soon as it has weight
        comp.state_cond[2].weight.normal_(std=0.02)
        assert not torch.allclose(comp(**kw, agent_state=st), ref), "state conditioning inert on trained weights"
    print(f"  warm start from {DYN_RUN.name[:24]}…: strict load + bit-identical forward on {dev}; conditioning live once weighted")


# ── §3.3 core protocol ──────────────────────────────────────────────────────
def _tiny_core(agent=True):
    from far.models.ldwm_core import LDWMCore
    from far.generators.diffusion import create_diffusion
    torch.manual_seed(0)
    gen = AgentStateCDiT(predictor_dim=32, predictor_heads=4, **SMALL_DIT) if agent else PoseFaCDiT(**SMALL_DIT)
    with torch.no_grad():                                           # non-trivial DiT output (see the composite test)
        dit = gen.dit if agent else gen
        dit.final_layer.linear.weight.normal_(std=0.02)
        for m in dit.modules():
            if hasattr(m, "adaLN_modulation"):
                m.adaLN_modulation[-1].weight.normal_(std=0.02)
    return LDWMCore(gen, create_diffusion(""), memory=None, agent_loss_weight=0.5, agent_reward_weight=1.0)


def _core_batch(B=2, T=6, G=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    st = r(B, T, 6); st[..., 0] = (st[..., 0] > 0).float(); st[..., 3] = (st[..., 3] > 0).float()
    return dict(context_latent=r(B, T, 4, 32, 32), goal_latent=r(B, G, 4, 32, 32), action=r(B, G, 5) * 0.1,
                stride=torch.rand(B, G, generator=g) * 3, context_poses=r(B, T, 5) * 0.1,
                context_times=torch.sort(torch.rand(B, T, generator=g) * 40, dim=1).values,
                context_mask=torch.ones(B, T, dtype=torch.bool), action_interact=torch.zeros(B, G, 4),
                action_state=torch.zeros(B, G, 4), context_agent_state=st,
                goal_agent_state=torch.cat([(r(B, G, 1) > 0).float(), r(B, G, 2)], -1))


def test_core_slot_gather():
    """`_slot_states` picks each selected time's nearest pool frame."""
    from far.models.ldwm_core import LDWMCore
    times = torch.tensor([[0.0, 1.0, 2.5, 4.0], [0.0, 0.5, 3.0, 9.0]])
    state = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
    sel = torch.tensor([[[1.1, 3.9], [2.4, 0.0]], [[8.0, 0.4], [3.1, 3.1]]])        # (B, 2, 2)
    out = LDWMCore._slot_states(state, times, sel)
    assert out.shape == (2, 2, 2, 6)
    assert torch.equal(out[0, 0, 0], state[0, 1]) and torch.equal(out[0, 0, 1], state[0, 3])
    assert torch.equal(out[0, 1, 0], state[0, 2]) and torch.equal(out[0, 1, 1], state[0, 0])
    assert torch.equal(out[1, 0, 0], state[1, 3]) and torch.equal(out[1, 0, 1], state[1, 1])
    print("  core: nearest-time slot gather OK")


def test_core_loss_and_sampling():
    """compute_loss_latent adds the predictor's loss, conditions the DiT on the
    (detached) prediction, and every parameter group receives gradient;
    sample_latent predicts once and exposes it; a generator without the
    protocol ignores the batch keys."""
    from far.generators.diffusion import create_diffusion
    core = _tiny_core(agent=True); b = _core_batch()
    opt_loss, gen_loss = core.compute_loss_latent(**b)
    assert torch.isfinite(opt_loss) and opt_loss.item() != gen_loss.item(), "predictor loss not added"
    assert float(core.last_agent_loss) > 0
    opt_loss.backward()
    gen = core.generator
    has = lambda mod: any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
    assert has(gen.predictor), "predictor got no gradient"
    assert has(gen.state_cond), "state conditioning got no gradient"
    assert has(gen.dit), "DiT got no gradient"
    # the conditioning link carries zero gradient into the predictor: with the
    # predictor loss weight at 0 its gradient must vanish exactly
    core0 = _tiny_core(agent=True); core0.agent_loss_weight = 0.0
    core0.compute_loss_latent(**b)[0].backward()
    assert not has(core0.generator.predictor), "detach is leaking generator gradient into the predictor"
    # the DiT really is conditioned on the prediction: perturbing the predictor's output changes the loss
    torch.manual_seed(1); ref = core.compute_loss_latent(**b)[1].item()
    with torch.no_grad():
        core.generator.state_cond[2].weight.normal_(std=0.05); core.generator.predictor.head[2].bias.add_(3.0)
    torch.manual_seed(1); alt = core.compute_loss_latent(**b)[1].item()
    assert alt != ref, "DiT output insensitive to the predicted state"
    # sampling: predict once, keep it
    core.diffusion = create_diffusion("ddim2"); core.eval()
    with torch.no_grad():
        out = core.sample_latent(b["context_latent"], b["action"].flatten(0, 1), b["stride"].flatten(0, 1), num_goals=2,
                                 context_poses=b["context_poses"], context_times=b["context_times"], context_mask=b["context_mask"],
                                 action_interact=b["action_interact"], action_state=b["action_state"], use_ddim=True,
                                 context_agent_state=b["context_agent_state"])
    assert out.shape == (4, 4, 32, 32) and core.last_agent_pred.shape == (4, 3)
    # no protocol on a bare generator: keys accepted and ignored
    bare = _tiny_core(agent=False)
    l0, l1 = bare.compute_loss_latent(**b)
    assert l0.item() == l1.item() and float(bare.last_agent_loss) == 0 and bare._agent_module() is None
    print("  core: loss + gradients + conditioning + sampling OK; bare generator unaffected")


# ── §3.6 retriever cue ──────────────────────────────────────────────────────
def _tiny_pnp(scoring="visual+meta_z", cue=True, chunk=3, num_sets=2):
    from far.memories.pnp_cl import PnPRetrieval, QueryViT
    torch.manual_seed(0)
    qv = QueryViT(img_height=32, img_width=32, patch_size=8, in_chans=4, embed_dim=32, depth=1,
                  num_heads=4, action_dim=5, key_dim=16)
    return PnPRetrieval(chunk_size=chunk, query_vit=qv, num_sets=num_sets, scoring=scoring,
                        metadata_hidden=32, metadata_layers=2, stride_to_seconds=1.0,
                        encoder_mode="frozen", metadata_agent_state=cue)


def _pnp_inputs(B=2, T=24, G=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    st = r(B, T, 6); st[..., 0] = (st[..., 0] > 0).float(); st[..., 3] = (st[..., 3] > 0).float()
    return dict(context_frames=r(B, T, 4, 32, 32), context_poses=r(B, T, 5) * 0.1,
                context_times=torch.sort(torch.rand(B, T, generator=g) * 40, dim=1).values,
                action=r(B, G, 5) * 0.1, stride=torch.rand(B, G, generator=g) * 3,
                context_mask=torch.ones(B, T, dtype=torch.bool), context_keys=r(B, T, 16)), st


def test_scorer_agent_cue():
    """The metadata scorer's agent block is a zero-init no-op, becomes live once
    weighted, and reads the pair velocity rotated by the frame's yaw."""
    from far.memories.pnp_cl import MetadataCrossScorer
    torch.manual_seed(0)
    sc = MetadataCrossScorer(hidden=32, num_layers=2, agent_state_dim=6).eval()
    with torch.no_grad():                                   # non-trivial scorer (its last layer is zero-init)
        sc.mlp[-1].weight.normal_(std=0.1)
    B, P, G = 2, 9, 2
    g = torch.Generator().manual_seed(1); r = lambda *s: torch.randn(*s, generator=g)
    poses, times = r(B, P, 5) * 0.1, torch.sort(torch.rand(B, P, generator=g) * 40, dim=1).values
    action, stride, cur = r(B, G, 5) * 0.1, torch.rand(B, G, generator=g) * 3, torch.full((B,), 40.0)
    st = r(B, P, 6); st[..., 0] = 1.0; st[..., 3] = 1.0
    with torch.no_grad():
        ref = sc(poses, times, action, stride, cur)
        assert torch.equal(sc(poses, times, action, stride, cur, agent_state=st), ref), "zero-init block is not a no-op"
        sc.agent_proj.weight.normal_(std=0.1)
        assert not torch.allclose(sc(poses, times, action, stride, cur, agent_state=st), ref), "cue block inert once weighted"
        # the feature block is gated on both paired observations being in view: a
        # partner-less frame contributes only its own state, so its velocity term is zero;
        # the CURRENT frame's own [vi, dx, dy] rides along as the query-side block (zeros when absent)
        feats = {}
        h = sc.agent_proj.register_forward_pre_hook(lambda m, i: feats.__setitem__("x", i[0].detach().clone()))
        st2 = st.clone(); st2[:, 0, 3] = 0.0; sc(poses, times, action, stride, cur, agent_state=st2)
        assert torch.equal(feats["x"][:, 0, 6:8], torch.zeros(B, 2)) and feats["x"].shape[-1] == 11
        assert torch.equal(feats["x"][..., 8:11], torch.zeros(B, P, 3))
        cs = torch.tensor([[1.0, 2.0, -0.5], [0.0, 0.0, 0.0]])
        with_cur = sc(poses, times, action, stride, cur, agent_state=st2, cur_state=cs); h.remove()
        assert torch.equal(feats["x"][0, :, 8:11], cs[0].expand(P, 3)), "current state not broadcast into the block"
        assert not torch.allclose(with_cur[0], sc(poses, times, action, stride, cur, agent_state=st2)[0]), "current state inert"
    assert MetadataCrossScorer(hidden=32, num_layers=2).agent_proj is None
    print("  scorer cue: zero-init no-op, live once weighted, velocity gated on the pair, current state on the query side")


def test_pnp_forward_with_cue():
    """`PnPRetrieval` accepts `context_agent_state`, slices it like the poses,
    returns identical selections at init, declares `uses_agent_state`, ignores
    the cue under visual scoring, and loads a cue-less checkpoint."""
    mem = _tiny_pnp().eval(); kw, st = _pnp_inputs()
    assert mem.uses_agent_state
    seen = {}
    h = mem.meta_scorer.register_forward_hook(
        lambda m, i, kwargs, o: seen.update(agent=kwargs.get("agent_state"), cur=kwargs.get("cur_state")), with_kwargs=True)
    with torch.no_grad():
        out0 = mem(**kw, num_memories=3)
        out1 = mem(**kw, num_memories=3, context_agent_state=st)
    h.remove()
    assert seen["agent"] is not None and seen["agent"].shape == (2, 21, 6), "cue not sliced into the scorer (T-1=23 -> drop 2 -> 7 chunks x 3)"
    assert seen["cur"] is not None and torch.equal(seen["cur"], st[:, -1, :3]), "current frame's state not passed as the query-side cue"
    assert torch.equal(out0[2], out1[2]) and torch.equal(out0[0], out1[0]), "zero-init cue changed the selection"
    # visual scoring: flag accepted, no scorer, kwarg ignored
    vis = _tiny_pnp(scoring="visual").eval()
    assert not vis.uses_agent_state
    with torch.no_grad():
        vis(**kw, num_memories=3, context_agent_state=st)
    # a checkpoint saved WITHOUT the block loads (expected-missing), and one WITH it round-trips
    cueless = _tiny_pnp(cue=False)
    torch.save({"memory": cueless.state_dict()}, "/tmp/_pnp_cueless.pth.tar")
    m2 = _tiny_pnp(); m2.load_weights("/tmp/_pnp_cueless.pth.tar")
    assert torch.equal(m2.meta_scorer.agent_proj.weight, torch.zeros_like(m2.meta_scorer.agent_proj.weight))
    torch.save({"memory": mem.state_dict()}, "/tmp/_pnp_cue.pth.tar")
    m3 = _tiny_pnp(); m3.load_weights("/tmp/_pnp_cue.pth.tar")
    assert all(torch.equal(a, b) for a, b in zip(m3.state_dict().values(), mem.state_dict().values()))
    print("  pnp forward: cue reaches the scorer chunk-aligned, selections unchanged at init, visual mode ignores it, checkpoints load")


def test_core_forwards_cue_to_memory():
    """`prepare_context` hands `context_agent_state` to a memory that declares
    `uses_agent_state` and withholds it otherwise."""
    from far.models.ldwm_core import LDWMCore
    from far.generators.diffusion import create_diffusion
    for cue in (True, False):
        mem = _tiny_pnp(cue=cue)
        gen = AgentStateCDiT(predictor_dim=32, predictor_heads=4, **{**SMALL_DIT, "context_size": 4})
        core = LDWMCore(gen, create_diffusion(""), memory=mem).eval()
        kw, st = _pnp_inputs()
        seen = {}
        h = mem.meta_scorer.register_forward_hook(lambda m, i, kwargs, o: seen.__setitem__("agent", kwargs.get("agent_state")), with_kwargs=True)
        b = _core_batch(B=2, T=24, G=2)
        b["context_agent_state"] = st; b["context_keys"] = kw["context_keys"]
        core.compute_loss_latent(**b)
        h.remove()
        assert (seen["agent"] is not None) == cue, f"cue forwarded={seen['agent'] is not None} but memory uses_agent_state={cue}"
    print("  core -> memory: cue forwarded iff the scorer carries the block")


def test_core_binarised_cond():
    """`LDWMCore._agent_cond`: teacher-forced ground truth in training, the predictor's
    decision when sampling, raw soft prediction when the knob is off; never a gradient
    path into the predictor beyond the zero-weight DDP link."""
    from types import SimpleNamespace
    from far.models.ldwm_core import LDWMCore
    pred = torch.tensor([[2.0, 0.3, -0.4], [-1.5, 0.7, 0.2], [0.1, -0.9, 0.5]], requires_grad=True)
    tgt = torch.tensor([[1.0, 1.2, -0.8], [0.0, 0.0, 0.0], [1.0, 0.4, 0.4]])
    core = SimpleNamespace(agent_cond_binary=True)
    train = LDWMCore._agent_cond(core, pred, tgt)
    assert torch.allclose(train.detach(), torch.tensor([[1.0, 1.2, -0.8], [-1.0, 0.0, 0.0], [1.0, 0.4, 0.4]])), train
    samp = LDWMCore._agent_cond(core, pred)
    assert torch.allclose(samp.detach(), torch.tensor([[1.0, 0.3, -0.4], [-1.0, 0.0, 0.0], [1.0, -0.9, 0.5]])), samp
    for out in (train, samp):
        assert out.requires_grad, "the zero-weight DDP link must keep the predictor reachable"
        g, = torch.autograd.grad(out.sum(), pred); assert torch.all(g == 0), "the link must carry exactly zero gradient"
    soft = LDWMCore._agent_cond(SimpleNamespace(agent_cond_binary=False), pred, tgt)
    assert torch.allclose(soft.detach(), pred.detach()), "knob off = the raw soft prediction"
    print("  binarised cond: ground truth in training, sign(logit) with zeroed position when sampling, soft when off; zero-gradient link")


def test_core_rag_select():
    """`rag_step(select=)`: all-True equals no selection bit-for-bit; all-False returns a x0 loss that
    still requires grad (the DDP collective must fire on every rank); a mixed mask runs the step on the
    selected rows only (visible in the reward telemetry)."""
    from far.models.ldwm_core import LDWMCore
    from far.generators.diffusion import create_diffusion
    torch.manual_seed(0)
    gen = AgentStateCDiT(predictor_dim=32, predictor_heads=4, **SMALL_DIT); mem = _tiny_pnp(cue=True, chunk=3, num_sets=2)
    core = LDWMCore(gen, create_diffusion(""), memory=mem, agent_loss_weight=0.5, agent_reward_weight=1.0)
    b = _core_batch(B=2, T=8, G=2)
    ema0 = {k: v.clone() for k, v in core.state_dict().items() if "tau_ema" in k}      # rag_step moves its τ EMAs
    def run(**kw):
        core.load_state_dict({**core.state_dict(), **ema0}); torch.manual_seed(1)
        return core.rag_step(**b, num_noise_levels=1, **kw)
    full = run()
    same = run(select=torch.tensor([True, True]))
    assert torch.allclose(full, same), "select=all-True must reproduce the unselected step"
    none = run(select=torch.tensor([False, False]))
    assert none.requires_grad and float(none) == 0.0, "no selected row must give a x0 loss that still carries the graph"
    part = run(select=torch.tensor([True, False]))
    assert part.requires_grad and torch.isfinite(part) and core.last_rag_terms["diffusion"].shape[0] == 1, "mixed select must run on the selected rows only"
    print("  rag_step(select): all-True == none; all-False -> x0 with graph; mixed -> selected rows only")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        import pytest
        skip_exc = pytest.skip.Exception          # `_skip` raises this when pytest is importable
    except ImportError:
        skip_exc = ()
    failed = skipped = 0
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except skip_exc as e:
            skipped += 1; print(f"  SKIP: {e}")
        except Exception as e:
            import traceback
            failed += 1; print(f"  FAIL: {type(e).__name__}: {e}"); traceback.print_exc()
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    sys.exit(1 if failed else 0)
