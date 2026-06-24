"""Warm, in-process HiDream-O1-Image-Dev (MLX) text-to-image engine.

The model ships with a one-shot CLI generator (`generate_hidream_o1_mlx.py`).
This module reuses that package's helper modules but loads the model ONCE and
keeps it resident, so a long-running server can answer many requests without
paying the load cost every time. The denoising loop is a faithful port of the
script's `run_inference` body.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATH = os.environ.get(
    "HIDREAM_MODEL",
    "/Users/joebains/.omlx/models/mlx-community/HiDream-O1-Image-Dev-mlx-bf16",
)
SCRIPTS_DIR = os.path.join(MODEL_PATH, "scripts", "hidream_o1")

# Device selection. The brand-new Apple M5 GPU was previously thought to have a
# broadly-broken MLX Metal backend (flat 32x32 colour-block images). Root cause
# was actually narrow: the M5 GPU matmul kernel uses tf32-style reduced-precision
# accumulation (~6e-4 rel error). Harmless for normal matmuls, but FATAL for the
# Qwen3-VL rotary embedding, which computes rope angles via a matmul
# (`inv_freq @ position_ids`). At image position ids up to ~4096, a 6e-4 relative
# error becomes ~2.5 radians of absolute error, and cos()/sin() amplify that into
# pure noise -> attention collapses -> DC-only per-patch output. The fix
# (`_patch_qwen3vl_rope` below) recomputes those angles with an element-wise
# broadcast multiply instead of a matmul, which stays in full fp32 on the GPU and
# matches CPU to ~4e-8. With that patch the entire forward runs correctly on the
# GPU (~10-20x faster than CPU). Default to GPU; set HIDREAM_DEVICE=cpu to fall
# back to the (slow but always-correct) CPU path.
DEVICE = os.environ.get("HIDREAM_DEVICE", "gpu").strip().lower()

_CTX = None            # cached (lazy) model context
_LOAD_ERROR = None     # last load failure, surfaced to /health


def _set_device(mx):
    """Pin MLX to the configured device (CPU by default — see DEVICE note)."""
    try:
        dev = mx.cpu if DEVICE in ("cpu", "") else mx.gpu
        mx.set_default_device(dev)
    except Exception:
        pass


def scripts_available() -> bool:
    return os.path.isfile(os.path.join(SCRIPTS_DIR, "generate_hidream_o1_mlx.py"))


def weights_ready() -> bool:
    """The backbone safetensors must be present (the big download) plus the
    custom heads. Returns False while the download is still in progress."""
    if not os.path.isfile(os.path.join(MODEL_PATH, "extras",
                                        "custom_heads.safetensors")):
        return False
    p = Path(MODEL_PATH)
    have_backbone = any(p.glob("*.safetensors")) or \
        any(p.glob("model*.safetensors"))
    # An in-progress hf download leaves a *.incomplete blob in .cache.
    incomplete = list((p / ".cache" / "huggingface" / "download").rglob(
        "*.incomplete")) if (p / ".cache").exists() else []
    return have_backbone and not incomplete


def _ensure_path():
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)


_ROPE_PATCHED = False


def _patch_qwen3vl_rope():
    """Make the Qwen3-VL rotary embedding numerically safe on the Apple M5 GPU.

    Upstream computes rope angles as `inv_freq_expanded @ position_ids_expanded`
    (a matmul whose contraction dim is 1, i.e. just an outer product). On the M5
    GPU the matmul path runs in tf32-style reduced precision (~6e-4 rel error);
    at image position ids up to ~4096 that is ~2.5 rad of absolute angle error,
    which cos()/sin() turn into noise and collapse attention. Replacing the
    matmul with an element-wise broadcast multiply keeps the computation in full
    fp32 on the GPU and matches the CPU result to ~4e-8. Mathematically identical
    to the original; only the kernel used to multiply changes.
    """
    global _ROPE_PATCHED
    if _ROPE_PATCHED:
        return
    import mlx.core as mx
    from mlx_vlm.models.qwen3_vl import language as _lang

    def _rope_call(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )
        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)
        # Element-wise broadcast multiply instead of `@` — avoids the M5 GPU
        # tf32 matmul path that corrupts large rope angles.
        freqs = inv_freq_expanded * position_ids_expanded
        freqs = mx.swapaxes(freqs, 2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)

    _lang.Qwen3VLRotaryEmbedding.__call__ = _rope_call
    _ROPE_PATCHED = True


def load():
    """Load the backbone + custom heads once and cache. Raises on failure."""
    global _CTX, _LOAD_ERROR
    if _CTX is not None:
        return _CTX
    _ensure_path()
    import mlx.core as mx
    from mlx_vlm import load as mlx_vlm_load
    from hidream_model import HiDreamConfig, build_model

    _set_device(mx)
    _patch_qwen3vl_rope()
    t0 = time.time()
    backbone, processor = mlx_vlm_load(MODEL_PATH)
    cfg = HiDreamConfig()
    model = build_model(cfg, backbone)
    custom_path = Path(MODEL_PATH) / "extras" / "custom_heads.safetensors"
    if not custom_path.exists():
        raise RuntimeError(f"missing custom heads: {custom_path}")
    custom_weights = mx.load(str(custom_path))
    model.load_weights(list(custom_weights.items()), strict=False)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") \
        else processor
    for n in ("boi", "bor", "eor", "bot", "tms"):
        if not hasattr(tokenizer, f"{n}_token"):
            setattr(tokenizer, f"{n}_token", f"<|{n}_token|>")

    _CTX = {"mx": mx, "backbone": backbone, "processor": processor,
            "model": model, "cfg": cfg, "tokenizer": tokenizer,
            "device": DEVICE,
            "load_seconds": round(time.time() - t0, 1)}
    _LOAD_ERROR = None
    return _CTX


def is_loaded() -> bool:
    return _CTX is not None


def generate(prompt: str, width: int = 1024, height: int = 1024,
             steps: int = 28, seed: int = 32, snap: bool = True,
             noise_scale: float = None, noise_clip_std: float = 2.5,
             blend_seams: int = 0, progress=None) -> np.ndarray:
    """Run text-to-image and return an HxWx3 uint8 RGB array."""
    _ensure_path()
    import mlx.core as mx
    from pipeline_helpers import (
        PATCH_SIZE, NOISE_SCALE_DEFAULT, T_EPS, build_attention_mask,
        find_closest_resolution, patchify, unpatchify, build_t2i_text_sample,
    )
    from flow_match import FlashFlowMatchScheduler, DEFAULT_TIMESTEPS
    from hidream_model import (forward_generation,
                               precompute_text_embeds_with_vision)

    _set_device(mx)
    ctx = load()
    model, cfg, tokenizer = ctx["model"], ctx["cfg"], ctx["tokenizer"]
    backbone = ctx["backbone"]
    if noise_scale is None:
        noise_scale = NOISE_SCALE_DEFAULT

    if snap:
        sw, sh = find_closest_resolution(width, height)
        width, height = sw, sh
    width = (width // PATCH_SIZE) * PATCH_SIZE
    height = (height // PATCH_SIZE) * PATCH_SIZE
    h_patches, w_patches = height // PATCH_SIZE, width // PATCH_SIZE

    sample = build_t2i_text_sample(prompt, height, width, tokenizer,
                                   ctx["processor"], backbone.config)
    input_ids = mx.array(sample["input_ids"])
    position_ids = mx.array(sample["position_ids"])
    token_types = mx.array(sample["token_types"])
    vinput_mask = sample["vinput_mask"]

    DTYPE_MIN = -1e4
    mask4d = mx.array(build_attention_mask(sample["token_types"],
                                           DTYPE_MIN)).astype(mx.bfloat16)

    rng_key = mx.random.key(seed + 1)
    noise = noise_scale * mx.random.normal((1, 3, height, width), key=rng_key)
    noise_np = np.asarray(noise)
    z = mx.array(patchify(noise_np[0])[None]).astype(mx.bfloat16)

    sched = FlashFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
    sched.set_timesteps(steps, custom_timesteps=DEFAULT_TIMESTEPS)
    noise_scale_schedule = np.linspace(noise_scale, noise_scale,
                                       len(sched.timesteps_np))

    vinput_idx = mx.array(np.where(vinput_mask[0])[0].astype(np.int32))
    tgt_idx = vinput_idx

    inputs_embeds_pre = precompute_text_embeds_with_vision(
        model, cfg, input_ids, pixel_values=None, image_grid_thw=None)
    mx.eval(inputs_embeds_pre)

    total = len(sched.timesteps_np)
    for step_idx, step_t in enumerate(sched.timesteps_np):
        t_pixeldit = mx.full([1], 1.0 - float(step_t) / 1000.0,
                             dtype=mx.float32)
        sigma = max(float(step_t) / 1000.0, T_EPS)
        x_pred = forward_generation(
            model, cfg, inputs_embeds_with_vision=inputs_embeds_pre,
            position_ids=position_ids, vinputs=z, timestep=t_pixeldit,
            input_ids=input_ids, token_types=token_types,
            attention_mask_4d=mask4d)
        gen_patches_mx = mx.take(x_pred, tgt_idx, axis=1).astype(mx.float32)
        v = (gen_patches_mx - z.astype(mx.float32)) / sigma
        z = sched.step(-v, float(step_t), z,
                       s_noise=float(noise_scale_schedule[step_idx]),
                       noise_clip_std=noise_clip_std, seed=seed)
        mx.eval(z)
        if progress:
            progress(step_idx + 1, total)

    img = (z + 1) / 2
    img_np = np.asarray(img[0].astype(mx.float32))
    rgb = unpatchify(img_np, h_patches, w_patches)
    arr = np.clip(rgb.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return arr, width, height
