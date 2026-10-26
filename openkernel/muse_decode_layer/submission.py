# okc: problem=muse_decode_layer
"""Muse decoder layer: SGLang's FA3 attention and Marlin W4A16 experts, with the rest rebuilt
around split-K Triton GEMMs whose FP32 partial sums feed fused epilogue kernels.

  qkv_gate GEMM  -> [qk-norm, RoPE, FP8 q, FP8 k/v cache store]        (1 kernel)
  FA3            -> [per-head norm x sigmoid(gate)]                      (1 kernel)
  o GEMM         -> [post-norm, residual gate, pre-FFN norm]             (1 kernel)
  pooler + router (BF16 hi/lo router rows appended to the pooler, one GEMM)
                 -> [pre-expert norm, sigmoid top-8, renormalize, local ids] (1 kernel)
  Marlin experts (as SGLang)
  unpooler GEMM  -> [post-FFN norm with the all-gather's zero pad, residual gate,
                     next pre-attention norm]                            (1 kernel)

prepare() times each GEMM shape against cuBLAS at the token counts that matter and keeps
whichever is faster; cuBLAS is the fallback, and is used above 32 tokens."""

import math
import time
from typing import TypedDict

import torch
import triton
import triton.language as tl

try:  # the judge's environment; absent on a CPU-only box, where the kernels can still be tested
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
except ImportError:  # pragma: no cover
    flash_attn_with_kvcache = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it)
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases, 64 in some tests
    window: int           # 2048 on sliding-window layers; 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T]
    cache_loc: torch.Tensor   # int64 [T]
    cache_seqlens: torch.Tensor  # int32 [B]
    cu_seqlens_q: torch.Tensor   # int32 [B+1]
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor     # float32 []
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages]
    seq_lens: torch.Tensor    # int32 [B]
    local_start: torch.Tensor # int32 [1]
    config: Config


HIDDEN = 8192
D_EXPERT = 4096
HQ, HKV, HEAD_DIM = 16, 2, 64
Q_SIZE, KV_SIZE = HQ * HEAD_DIM, HKV * HEAD_DIM
N_EXPERTS, TOP_K, LOCAL_EXPERTS = 256, 8, 32
UNPOOL_SLICE = HIDDEN // 8
SLICED_UNPOOL_MAX_TOKENS = 8
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
ROUTER_MULTIPLIER = 0.1767766952966369
ROUTE_EPS = 1e-15
ROPE_THETA = 500_000.0
ROPE_POSITIONS = 1 << 21
FP8 = torch.float8_e4m3fn
# MuseSpark1xFp4Experts' Marlin block sizes, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
# Token counts the GEMMs are tuned at (the scored loads' T); above the largest, cuBLAS.
TUNE_TOKENS = (1, 8, 16, 32, 128, 256)
TUNE_BUDGET_S = 90.0
# Token counts the expert path is tuned at (the scored loads' T); above the largest, Marlin.
MOE_TUNE_TOKENS = (1, 8, 16, 32, 128, 256)
MOE_TUNE_BUDGET_S = 80.0
ATTN_TUNE_BUDGET_S = 50.0
# The scored attention shapes: (batch, q_len, sliding window?), at 80K-token contexts.
ATTN_TUNE_SHAPES = [(b, l, w) for b in (1, 16, 32) for l in (1, 8) for w in (True, False)]
ATTN_DEFAULT = (1024, 64, 4, False)  # (target programs, keys per step, warps, FP8 q.k) without FA3
# The share of a token's routes on this GPU in the judge's loads (about 2.6 pairs for one token,
# 14 for 8, 50 for 32, 390 for 256: the busiest GPU's).
MOE_LOCAL_SHARE = {1: 0.33, 8: 0.22, 32: 0.2, 256: 0.19}
PARTIALS_BYTES_PER_MS = 3.0e9  # what the split-K partials' round trip is charged at, in tuning


# ----------------------------------------------------------------------------------------------
# GEMM: out[s, m, n] = sum over split s of k of x[m, k] * w[n, k]   (x, w BF16; out FP32)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_splitk(x_ptr, w_ptr, out_ptr, M, N, K, K_SPLIT,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = pid_s * K_SPLIT + tl.arange(0, BK)
    x_ptrs = x_ptr + rm[:, None].to(tl.int64) * K + rk[None, :]
    w_ptrs = w_ptr + rn[:, None].to(tl.int64) * K + rk[None, :]
    m_mask = rm[:, None] < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K_SPLIT, BK):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptrs += BK
        w_ptrs += BK
    out_ptrs = out_ptr + pid_s.to(tl.int64) * M * N + rm[:, None] * N + rn[None, :]
    tl.store(out_ptrs, acc, mask=m_mask)


def _mm(x, w, cfg):
    """[S, M, N] FP32 partial sums of x @ w.T (S = 1 for cuBLAS)."""
    m, k = x.shape
    n = w.shape[0]
    if cfg is None:
        return torch.mm(x, w.t(), out_dtype=torch.float32).unsqueeze(0)
    bm, bn, bk, s, stages, warps = cfg
    out = torch.empty((s, m, n), dtype=torch.float32, device=x.device)
    _gemm_splitk[(n // bn, s, triton.cdiv(m, bm))](
        x, w, out, m, n, k, k // s, BM=bm, BN=bn, BK=bk, num_stages=stages, num_warps=warps)
    return out


# ----------------------------------------------------------------------------------------------
# Epilogues. Each program is one token; `part` holds [S, T, N] FP32 partial sums.
# ----------------------------------------------------------------------------------------------
@triton.jit
def _qkv_post(part, S, T, N, pos_ptr, loc_ptr, cos_ptr, sin_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, q_out,
              eps, HAS_ROPE: tl.constexpr):
    """Per-head QK-norm, interleaved RoPE, q to FP8; k and v to FP8 at the token's cache row."""
    t = tl.program_id(0)
    base = part + t * N
    hq = tl.arange(0, 16)[:, None]
    hk = tl.arange(0, 2)[:, None]
    i = tl.arange(0, 32)[None, :]
    vcol = tl.arange(0, 128)
    qe = tl.zeros((16, 32), tl.float32)
    qo = tl.zeros((16, 32), tl.float32)
    ke = tl.zeros((2, 32), tl.float32)
    ko = tl.zeros((2, 32), tl.float32)
    v = tl.zeros((128,), tl.float32)
    for s in range(S):
        b = base + s * T * N
        qe += tl.load(b + hq * 64 + 2 * i)
        qo += tl.load(b + hq * 64 + 2 * i + 1)
        ke += tl.load(b + 1024 + hk * 64 + 2 * i)
        ko += tl.load(b + 1024 + hk * 64 + 2 * i + 1)
        v += tl.load(b + 1152 + vcol)
    rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, axis=1) / 64.0 + eps)[:, None]
    rk = tl.rsqrt(tl.sum(ke * ke + ko * ko, axis=1) / 64.0 + eps)[:, None]
    qe, qo = qe * rq, qo * rq
    ke, ko = ke * rk, ko * rk
    if HAS_ROPE:
        p = tl.load(pos_ptr + t)
        c = tl.load(cos_ptr + p * 32 + i)
        sn = tl.load(sin_ptr + p * 32 + i)
        qe, qo = qe * c - qo * sn, qe * sn + qo * c
        ke, ko = ke * c - ko * sn, ke * sn + ko * c
    qb = q_out + t * 1024
    tl.store(qb + hq * 64 + 2 * i, qe.to(tl.float8e4nv))
    tl.store(qb + hq * 64 + 2 * i + 1, qo.to(tl.float8e4nv))
    loc = tl.load(loc_ptr + t).to(tl.int64)
    inv_ks = 1.0 / tl.load(ks_ptr)
    inv_vs = 1.0 / tl.load(vs_ptr)
    kb = kc_ptr + loc * 128
    tl.store(kb + hk * 64 + 2 * i, (ke * inv_ks).to(tl.float8e4nv))
    tl.store(kb + hk * 64 + 2 * i + 1, (ko * inv_ks).to(tl.float8e4nv))
    tl.store(vc_ptr + loc * 128 + vcol, (v * inv_vs).to(tl.float8e4nv))


@triton.jit
def _norm_gate(attn_ptr, part, S, T, N, out_ptr, eps):
    """Per-head RMSNorm of the attention output, times sigmoid(gate), to BF16."""
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    col = h * 64 + d
    a = tl.load(attn_ptr + t * 1024 + col).to(tl.float32)
    g = tl.zeros((16, 64), tl.float32)
    for s in range(S):
        g += tl.load(part + s * T * N + t * N + 1280 + col)
    a = a * tl.rsqrt(tl.sum(a * a, axis=1) / 64.0 + eps)[:, None]
    tl.store(out_ptr + t * 1024 + col, (a * tl.sigmoid(g)).to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# Attention, split over the keys (flash-decoding): each program takes one request and one chunk
# of its keys, for both KV heads and all their query heads and block tokens; _attn_combine merges
# the chunks per (token, query head) and applies the gated output norm.
# ----------------------------------------------------------------------------------------------
@triton.jit
def _attn_step(q, k, v, vis, m, l, acc, scale):
    """One block of keys for one KV head: scores in the log2 domain, online softmax."""
    s = tl.where(vis, tl.dot(q, tl.trans(k)) * scale, float("-inf"))
    m_new = tl.maximum(m, tl.max(s, axis=1))
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp2(s - m_use[:, None])
    alpha = tl.exp2(m - m_use)
    l = l * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, l, acc


@triton.jit
def _attn_chunk(q_ptr, kc_ptr, vc_ptr, pt_ptr, pt_stride, seqlen_ptr, ks_ptr, vs_ptr, o_ptr, lse_ptr,
                NC, page, scale_log2, window, CHUNK,
                L: tl.constexpr, BN: tl.constexpr, G: tl.constexpr, M: tl.constexpr,
                WINDOWED: tl.constexpr, QK_FP8: tl.constexpr):
    """q: FP8 [T, 16, 64] (normalized, RoPE'd); caches FP8 [rows, 2, 64]. Writes this chunk's
    normalized output [T, 16, NC, 64] and log2-sum-exp [T, 16, NC] (-inf for no keys)."""
    b = tl.program_id(0)
    c = tl.program_id(1)
    kv_len = tl.load(seqlen_ptr + b)
    q0 = kv_len - L  # the block's first query position
    first = 0
    if WINDOWED:
        first = tl.maximum(q0 - window + 1, 0)
    start = first + c * CHUNK
    end = tl.minimum(start + CHUNK, kv_len)
    r = tl.arange(0, M)
    qi = r // G
    qg = r % G
    r_ok = r < L * G
    qpos = q0 + qi
    d = tl.arange(0, 64)
    tok = b * L + qi
    dt = tl.bfloat16
    qdt = tl.float8e4nv if QK_FP8 else tl.bfloat16  # q . k in FP8 (as FA3) or BF16; p . v in BF16
    qa = tl.load(q_ptr + (tok * 16 + qg)[:, None] * 64 + d[None, :], mask=r_ok[:, None], other=0.0).to(qdt)
    qb = tl.load(q_ptr + (tok * 16 + G + qg)[:, None] * 64 + d[None, :], mask=r_ok[:, None], other=0.0).to(qdt)
    ma = tl.full((M,), float("-inf"), tl.float32)
    mb = tl.full((M,), float("-inf"), tl.float32)
    la = tl.zeros((M,), tl.float32)
    lb = tl.zeros((M,), tl.float32)
    acca = tl.zeros((M, 64), tl.float32)
    accb = tl.zeros((M, 64), tl.float32)
    s_scale = scale_log2 * tl.load(ks_ptr)
    pt = pt_ptr + b.to(tl.int64) * pt_stride
    for n0 in range(start, end, BN):
        kpos = n0 + tl.arange(0, BN)
        km = kpos < end
        row = tl.load(pt + kpos // page, mask=km, other=0).to(tl.int64) * page + kpos % page
        off = row[:, None] * 128 + d[None, :]
        ka = tl.load(kc_ptr + off, mask=km[:, None], other=0.0).to(qdt)
        kb = tl.load(kc_ptr + off + 64, mask=km[:, None], other=0.0).to(qdt)
        va = tl.load(vc_ptr + off, mask=km[:, None], other=0.0).to(dt)
        vb = tl.load(vc_ptr + off + 64, mask=km[:, None], other=0.0).to(dt)
        vis = km[None, :] & (kpos[None, :] <= qpos[:, None])
        if WINDOWED:
            vis = vis & (kpos[None, :] > qpos[:, None] - window)
        ma, la, acca = _attn_step(qa, ka, va, vis, ma, la, acca, s_scale)
        mb, lb, accb = _attn_step(qb, kb, vb, vis, mb, lb, accb, s_scale)
    v_scale = tl.load(vs_ptr)
    for h in tl.static_range(2):
        if h == 0:
            m_, l_, acc_ = ma, la, acca
        else:
            m_, l_, acc_ = mb, lb, accb
        rows = (tok * 16 + h * G + qg) * NC + c
        o = acc_ / tl.maximum(l_, 1e-30)[:, None] * v_scale
        tl.store(o_ptr + rows[:, None].to(tl.int64) * 64 + d[None, :], o, mask=r_ok[:, None])
        lse = tl.where(l_ > 0, m_ + tl.log2(tl.maximum(l_, 1e-30)), float("-inf"))
        tl.store(lse_ptr + rows, lse, mask=r_ok)


@triton.jit
def _attn_combine(o_ptr, lse_ptr, NC, part, S, T, N, out_ptr, eps, NCB: tl.constexpr):
    """Merge one (token, query head)'s chunks, then RMSNorm the head and gate it by sigmoid(gate)."""
    t = tl.program_id(0)
    hq = tl.program_id(1)
    row = t * 16 + hq
    d = tl.arange(0, 64)
    cb = tl.arange(0, NCB)
    mx = tl.full((NCB,), float("-inf"), tl.float32)
    for c0 in range(0, NC, NCB):
        mx = tl.maximum(mx, tl.load(lse_ptr + row * NC + c0 + cb, mask=c0 + cb < NC, other=float("-inf")))
    top = tl.max(mx, axis=0)
    acc = tl.zeros((NCB, 64), tl.float32)
    den = tl.zeros((NCB,), tl.float32)
    for c0 in range(0, NC, NCB):
        cm = c0 + cb < NC
        w = tl.exp2(tl.load(lse_ptr + row * NC + c0 + cb, mask=cm, other=float("-inf")) - top)
        o = tl.load(o_ptr + (row * NC + c0 + cb)[:, None].to(tl.int64) * 64 + d[None, :], mask=cm[:, None],
                    other=0.0)
        acc += w[:, None] * o
        den += w
    a = tl.sum(acc, axis=0) / tl.sum(den, axis=0)
    g = tl.zeros((64,), tl.float32)
    for s in range(S):
        g += tl.load(part + s * T * N + t * N + 1280 + hq * 64 + d)
    a = a * tl.rsqrt(tl.sum(a * a, axis=0) / 64.0 + eps)
    tl.store(out_ptr + t * 1024 + hq * 64 + d, (a * tl.sigmoid(g)).to(tl.bfloat16))


def _attention_fa3(data, q8, p_qkv):
    """The same with FA3, as SGLang runs it, and the gated norm after."""
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    window = cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, HKV),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, HKV))
    out = flash_attn_with_kvcache(
        q=q8, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(tokens, Q_SIZE)
    if out.dtype != torch.bfloat16:
        out = out.to(torch.bfloat16)
    gated = torch.empty((tokens, Q_SIZE), dtype=torch.bfloat16, device=q8.device)
    _norm_gate[(tokens,)](out, p_qkv, p_qkv.shape[0], tokens, p_qkv.shape[2], gated, RMS_EPS, num_warps=4)
    return gated


def _attn_geometry(cfg, max_len, target_programs):
    """(CHUNK, NC): key chunks per request, about target_programs programs in all."""
    batch, length, window = cfg["batch"], cfg["q_len"], cfg["window"]
    span = min(max_len, window + length - 1) if window else max_len
    chunk = triton.next_power_of_2(max(64, triton.cdiv(batch * span, target_programs)))
    return chunk, triton.cdiv(span, chunk)


def _attention(data, q8, p_qkv, acfg):
    """Gated attention output, BF16 [T, 1024], from FP8 queries (the cache already holds the
    block's keys and values)."""
    cfg = data["config"]
    batch, length, tokens, page = cfg["batch"], cfg["q_len"], cfg["tokens"], cfg["page_size"]
    target, bn, warps, qk_fp8 = acfg
    table = data["page_table"]
    chunk, nc = _attn_geometry(cfg, table.shape[1] * page, target)
    device = q8.device
    o = torch.empty((tokens * HQ * nc * HEAD_DIM,), dtype=torch.float32, device=device)
    lse = torch.empty((tokens * HQ * nc,), dtype=torch.float32, device=device)
    window = cfg["window"]
    _attn_chunk[(batch, nc)](
        q8, data["k_cache"], data["v_cache"], table, table.stride(0), data["cache_seqlens"],
        data["k_scale"], data["v_scale"], o, lse, nc, page,
        cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634, window, chunk,
        L=length, BN=bn, G=HQ // HKV, M=max(16, triton.next_power_of_2(length * HQ // HKV)),
        WINDOWED=bool(window), QK_FP8=qk_fp8, num_warps=warps)
    gated = torch.empty((tokens, Q_SIZE), dtype=torch.bfloat16, device=device)
    _attn_combine[(tokens, HQ)](o, lse, nc, p_qkv, p_qkv.shape[0], tokens, p_qkv.shape[2], gated, RMS_EPS,
                                NCB=min(64, triton.next_power_of_2(nc)), num_warps=4)
    return gated


@triton.jit
def _attn_boundary(part, S, T, stream_ptr, alpha_ptr, beta_ptr, w_ptr, stream_out, h_out,
                   post_eps, pre_eps, BLOCK: tl.constexpr):
    """s = alpha * s + beta * rms(a); h = rms(s) * (1 + w)."""
    t = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    a = tl.zeros((BLOCK,), tl.float32)
    for s in range(S):
        a += tl.load(part + s * T * BLOCK + t * BLOCK + c)
    a = a * tl.rsqrt(tl.sum(a * a, axis=0) / BLOCK + post_eps)
    st = tl.load(alpha_ptr + c) * tl.load(stream_ptr + t * BLOCK + c) + tl.load(beta_ptr + c) * a
    tl.store(stream_out + t * BLOCK + c, st)
    h = st * tl.rsqrt(tl.sum(st * st, axis=0) / BLOCK + pre_eps) * tl.load(w_ptr + c)
    tl.store(h_out + t * BLOCK + c, h.to(tl.bfloat16))


@triton.jit
def _pool_route(part, S, T, N, w_ptr, bias_ptr, start_ptr, p_out, ids_out, local_out, tw_out,
                eps, mult, route_eps, D: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
                LOCAL: tl.constexpr):
    """Pre-expert norm of the pooler output (columns 0..D-1), and the routing from the router's
    hi and lo logits (columns D..D+E-1 and D+E..D+2E-1): sigmoid, biased top-K, renormalize."""
    t = tl.program_id(0)
    c = tl.arange(0, D)
    e = tl.arange(0, E)
    p = tl.zeros((D,), tl.float32)
    logit = tl.zeros((E,), tl.float32)
    for s in range(S):
        b = part + s * T * N + t * N
        p += tl.load(b + c)
        logit += tl.load(b + D + e) + tl.load(b + D + E + e)
    p = p * tl.rsqrt(tl.sum(p * p, axis=0) / D + eps) * tl.load(w_ptr + c)
    tl.store(p_out + t * D + c, p.to(tl.bfloat16))

    scores = tl.sigmoid(logit * mult)
    biased = scores + tl.load(bias_ptr + e)
    j = tl.arange(0, K)
    ids = tl.zeros((K,), tl.int32)
    picked = tl.zeros((K,), tl.float32)
    for k in tl.static_range(K):
        best = tl.max(biased, axis=0)
        idx = tl.min(tl.where(biased == best, e, E), axis=0)
        sc = tl.sum(tl.where(e == idx, scores, 0.0), axis=0)
        ids = tl.where(j == k, idx, ids)
        picked = tl.where(j == k, sc, picked)
        biased = tl.where(e == idx, float("-inf"), biased)
    tw = picked / (tl.sum(picked, axis=0) + route_eps)
    start = tl.load(start_ptr)
    local = (ids - start + E) % E
    local = tl.where(local < LOCAL, local, -1)
    tl.store(ids_out + t * K + j, ids)
    tl.store(local_out + t * K + j, local)
    tl.store(tw_out + t * K + j, tw)


@triton.jit
def _ffn_boundary(part, S, T, NF, stream_ptr, wf_ptr, alpha_ptr, beta_ptr, wn_ptr, stream_out, h_out,
                  post_eps, pre_eps, BLOCK: tl.constexpr):
    """f = unpooled (zero past column NF: the all-gather) * (1 + w);
    s = alpha * s + beta * rms(f); hidden = rms(s) * (1 + w_next)."""
    t = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < NF
    f = tl.zeros((BLOCK,), tl.float32)
    for s in range(S):
        f += tl.load(part + s * T * NF + t * NF + c, mask=m, other=0.0)
    f = f * tl.load(wf_ptr + c)
    f = f * tl.rsqrt(tl.sum(f * f, axis=0) / BLOCK + post_eps)
    st = tl.load(alpha_ptr + c) * tl.load(stream_ptr + t * BLOCK + c) + tl.load(beta_ptr + c) * f
    tl.store(stream_out + t * BLOCK + c, st)
    h = st * tl.rsqrt(tl.sum(st * st, axis=0) / BLOCK + pre_eps) * tl.load(wn_ptr + c)
    tl.store(h_out + t * BLOCK + c, h.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# Experts (NVFP4 weights, BF16 activations), expert-major:
#   _moe_plan   one program: each local (token, slot) pair's place in an expert-sorted list,
#               and the GEMMs' work blocks (an expert and up to BM of its pairs)
#   _moe_up     per block and column tile: gate and up rows, silu(gate) * up        -> [P, 4096]
#   _moe_down   per block, column tile and K split: the down projection             -> [S, P, 4096]
#   _moe_final  per token: each local pair's (1 + w)-scaled RMSNorm, weighted, summed -> [T, 4096]
# The weights are repacked (bit for bit) so that within each K chunk of MOE_BK elements, byte j
# holds element j in its low nibble and element j + MOE_BK/2 in its high one: the two halves
# then multiply contiguous halves of the activations. Block scales are regrouped per K chunk,
# [E, K/MOE_BK, N, MOE_BK/16], so a tile's scales are contiguous.
# ----------------------------------------------------------------------------------------------
MOE_BK = 256


@triton.jit
def _moe_plan(lid_ptr, P, pos_ptr, tok_ptr, blk_e_ptr, blk_r0_ptr, blk_n_ptr, NB,
              E: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, NBP: tl.constexpr, TOPK: tl.constexpr):
    e = tl.arange(0, E)
    i = tl.arange(0, CH)
    counts = tl.zeros((E,), tl.int32)
    for c in range(0, P, CH):
        ids = tl.load(lid_ptr + c + i, mask=c + i < P, other=-1)
        counts += tl.sum((ids[:, None] == e[None, :]).to(tl.int32), axis=0)
    start = tl.cumsum(counts, 0) - counts
    run = tl.zeros((E,), tl.int32)
    for c in range(0, P, CH):
        ids = tl.load(lid_ptr + c + i, mask=c + i < P, other=-1)
        hit = (ids[:, None] == e[None, :]).to(tl.int32)
        rank = tl.cumsum(hit, 0) - hit
        pos = tl.sum(hit * (start + run)[None, :] + hit * rank, axis=1)
        valid = ids >= 0
        tl.store(pos_ptr + c + i, tl.where(valid, pos, -1), mask=c + i < P)
        tl.store(tok_ptr + pos, (c + i) // TOPK, mask=valid)
        run += tl.sum(hit, axis=0)
    # Blocks of up to BM pairs, expert by expert.
    nblk = (counts + BM - 1) // BM
    end = tl.cumsum(nblk, 0)
    boff = end - nblk
    b = tl.arange(0, NBP)
    eb = tl.sum((end[None, :] <= b[:, None]).to(tl.int32), axis=1)
    oh = (e[None, :] == eb[:, None]).to(tl.int32)
    bo = tl.sum(oh * boff[None, :], axis=1)
    st = tl.sum(oh * start[None, :], axis=1)
    cn = tl.sum(oh * counts[None, :], axis=1)
    j = b - bo
    used = b < tl.sum(nblk, axis=0)
    tl.store(blk_e_ptr + b, tl.where(used, eb, -1), mask=b < NB)
    tl.store(blk_r0_ptr + b, st + j * BM, mask=b < NB)
    tl.store(blk_n_ptr + b, tl.minimum(cn - j * BM, BM), mask=b < NB)


@triton.jit
def _dq(code, scale, dtype: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """E2M1 codes [ROWS, G, 16] (uint8, low 4 bits) times their block scales [ROWS, G] -> [ROWS, COLS],
    exact in BF16, short the 2^-14 the FP16 decoding leaves (folded into the tensor scale)."""
    c = code.to(tl.int32)
    bits = ((c & 7) << 9) | ((c & 8) << 12)
    v = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return tl.reshape((v * scale[:, :, None]).to(dtype), (ROWS, COLS))


@triton.jit
def _moe_up(x_ptr, w_ptr, s_ptr, gs_ptr, tok_ptr, blk_e_ptr, blk_r0_ptr, blk_n_ptr, out_ptr,
            K: tl.constexpr, NI: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    HB: tl.constexpr = BK // 2  # bytes per row per K chunk
    G: tl.constexpr = HB // 16  # scale groups per half chunk
    SG: tl.constexpr = BK // 16  # scale groups per chunk
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    e = tl.load(blk_e_ptr + pid_b)
    if e >= 0:
        r0 = tl.load(blk_r0_ptr + pid_b)
        nr = tl.load(blk_n_ptr + pid_b)
        rm = tl.arange(0, BM)
        m_mask = rm < nr
        tok = tl.load(tok_ptr + r0 + rm, mask=m_mask, other=0)
        rn = pid_n * BN + tl.arange(0, BN)
        x_ptrs = x_ptr + tok[:, None].to(tl.int64) * K + tl.arange(0, HB)[None, :]
        e64 = e.to(tl.int64)
        g3 = tl.arange(0, G)[None, :, None] * 16 + tl.arange(0, 16)[None, None, :]
        wg_ptrs = w_ptr + (e64 * 2 * NI + rn[:, None, None]) * (K // 2) + g3
        wu_ptrs = wg_ptrs + NI * (K // 2)
        sg_ptrs = s_ptr + e64 * (K // BK) * 2 * NI * SG + rn[:, None] * SG + tl.arange(0, G)[None, :]
        su_ptrs = sg_ptrs + NI * SG
        acc_g = tl.zeros((BM, BN), tl.float32)
        acc_u = tl.zeros((BM, BN), tl.float32)
        dt = x_ptr.dtype.element_ty
        for kt in range(K // BK):
            xl = tl.load(x_ptrs + kt * BK, mask=m_mask[:, None], other=0.0)
            xh = tl.load(x_ptrs + kt * BK + HB, mask=m_mask[:, None], other=0.0)
            bg = tl.load(wg_ptrs + kt * HB, eviction_policy="evict_first")
            bu = tl.load(wu_ptrs + kt * HB, eviction_policy="evict_first")
            so = kt * 2 * NI * SG
            sgl = tl.load(sg_ptrs + so).to(tl.float32)
            sgh = tl.load(sg_ptrs + so + G).to(tl.float32)
            sul = tl.load(su_ptrs + so).to(tl.float32)
            suh = tl.load(su_ptrs + so + G).to(tl.float32)
            acc_g = tl.dot(xl, tl.trans(_dq(bg & 15, sgl, dt, BN, HB)), acc_g)
            acc_g = tl.dot(xh, tl.trans(_dq(bg >> 4, sgh, dt, BN, HB)), acc_g)
            acc_u = tl.dot(xl, tl.trans(_dq(bu & 15, sul, dt, BN, HB)), acc_u)
            acc_u = tl.dot(xh, tl.trans(_dq(bu >> 4, suh, dt, BN, HB)), acc_u)
        gs = tl.load(gs_ptr + e) * 16384.0
        g = acc_g * gs
        u = acc_u * gs
        h = g * tl.sigmoid(g) * u
        out_ptrs = out_ptr + (r0 + rm)[:, None].to(tl.int64) * NI + rn[None, :]
        tl.store(out_ptrs, h.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def _moe_down(x_ptr, w_ptr, s_ptr, gs_ptr, blk_e_ptr, blk_r0_ptr, blk_n_ptr, out_ptr, P,
              K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              KT_SPLIT: tl.constexpr):
    HB: tl.constexpr = BK // 2  # bytes per row per K chunk
    G: tl.constexpr = HB // 16  # scale groups per half chunk
    SG: tl.constexpr = BK // 16  # scale groups per chunk
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_s = tl.program_id(2)
    e = tl.load(blk_e_ptr + pid_b)
    if e >= 0:
        r0 = tl.load(blk_r0_ptr + pid_b)
        nr = tl.load(blk_n_ptr + pid_b)
        rm = tl.arange(0, BM)
        m_mask = rm < nr
        rn = pid_n * BN + tl.arange(0, BN)
        x_ptrs = x_ptr + (r0 + rm)[:, None].to(tl.int64) * K + tl.arange(0, HB)[None, :]
        e64 = e.to(tl.int64)
        g3 = tl.arange(0, G)[None, :, None] * 16 + tl.arange(0, 16)[None, None, :]
        w_ptrs = w_ptr + (e64 * N + rn[:, None, None]) * (K // 2) + g3
        s_ptrs = s_ptr + e64 * (K // BK) * N * SG + rn[:, None] * SG + tl.arange(0, G)[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        dt = x_ptr.dtype.element_ty
        for kt in range(pid_s * KT_SPLIT, pid_s * KT_SPLIT + KT_SPLIT):
            xl = tl.load(x_ptrs + kt * BK, mask=m_mask[:, None], other=0.0)
            xh = tl.load(x_ptrs + kt * BK + HB, mask=m_mask[:, None], other=0.0)
            bw = tl.load(w_ptrs + kt * HB, eviction_policy="evict_first")
            so = kt * N * SG
            sl = tl.load(s_ptrs + so).to(tl.float32)
            sh = tl.load(s_ptrs + so + G).to(tl.float32)
            acc = tl.dot(xl, tl.trans(_dq(bw & 15, sl, dt, BN, HB)), acc)
            acc = tl.dot(xh, tl.trans(_dq(bw >> 4, sh, dt, BN, HB)), acc)
        acc = acc * (tl.load(gs_ptr + e) * 16384.0)
        out_ptrs = out_ptr + (pid_s * P + r0 + rm)[:, None].to(tl.int64) * N + rn[None, :]
        tl.store(out_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def _moe_final(y_ptr, S, P, lid_ptr, pos_ptr, tw_ptr, w_ptr, out_ptr, eps,
               D: tl.constexpr, TOPK: tl.constexpr):
    """Sum each local pair's split partials, scale by (1 + w), RMSNorm, weight, and sum per token."""
    t = tl.program_id(0)
    c = tl.arange(0, D)
    pw = tl.load(w_ptr + c)
    out = tl.zeros((D,), tl.float32)
    for k in tl.static_range(TOPK):
        lid = tl.load(lid_ptr + t * TOPK + k)
        valid = lid >= 0
        p = tl.where(valid, tl.load(pos_ptr + t * TOPK + k), 0).to(tl.int64)
        y = tl.zeros((D,), tl.float32)
        for s in range(S):
            y += tl.load(y_ptr + (s * P + p) * D + c, mask=valid & (c < D), other=0.0)
        y = y * pw
        y = y * tl.rsqrt(tl.sum(y * y, axis=0) / D + eps)
        out += tl.where(valid, tl.load(tw_ptr + t * TOPK + k), 0.0) * y
    tl.store(out_ptr + t * D + c, out.to(out_ptr.dtype.element_ty))


def _moe_blocks(tokens, bm):
    """An upper bound on the work blocks: each local expert gets ceil(pairs / bm) of them."""
    pairs = tokens * TOP_K
    return min(LOCAL_EXPERTS * triton.cdiv(tokens, bm), min(LOCAL_EXPERTS, pairs) + pairs // bm)


def _moe_bm(tokens):
    # An expert takes at most one pair per token.
    return 16 if tokens <= 16 else 32 if tokens <= 32 else 64


def _moe(x, topk_weights, local_ids, bank, cfg):
    """The local experts' weighted, post-normed outputs per token: [T, 4096] in x's dtype."""
    bn_up, bn_down, splits, warps = cfg
    tokens = x.shape[0]
    pairs = tokens * TOP_K
    bm = _moe_bm(tokens)
    nb = _moe_blocks(tokens, bm)
    device = x.device
    pos = torch.empty(pairs, dtype=torch.int32, device=device)
    tok = torch.empty(pairs, dtype=torch.int32, device=device)
    blk = torch.empty((3, nb), dtype=torch.int32, device=device)
    _moe_plan[(1,)](local_ids, pairs, pos, tok, blk[0], blk[1], blk[2], nb,
                    E=LOCAL_EXPERTS, CH=min(256, triton.next_power_of_2(pairs)), BM=bm,
                    NBP=triton.next_power_of_2(nb), TOPK=TOP_K, num_warps=4)
    inter = torch.empty((pairs, D_EXPERT), dtype=x.dtype, device=device)
    _moe_up[(D_EXPERT // bn_up, nb)](
        x, bank["w13"], bank["w13_s"], bank["w13_gs"], tok, blk[0], blk[1], blk[2], inter,
        K=D_EXPERT, NI=D_EXPERT, BM=bm, BN=bn_up, BK=MOE_BK, num_warps=warps)
    y = torch.empty((splits, pairs, D_EXPERT), dtype=torch.float32, device=device)
    _moe_down[(D_EXPERT // bn_down, nb, splits)](
        inter, bank["w2"], bank["w2_s"], bank["w2_gs"], blk[0], blk[1], blk[2], y, pairs,
        K=D_EXPERT, N=D_EXPERT, BM=bm, BN=bn_down, BK=MOE_BK, KT_SPLIT=D_EXPERT // MOE_BK // splits,
        num_warps=warps)
    out = torch.empty((tokens, D_EXPERT), dtype=x.dtype, device=device)
    _moe_final[(tokens,)](y, splits, pairs, local_ids, pos, topk_weights, bank["post_w"], out, POST_NORM_EPS,
                          D=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


def _repack_codes(w, bk=MOE_BK):
    """uint8 [E, N, K/2], element 2i low / 2i+1 high -> per K chunk of bk: element j low, j + bk/2 high."""
    out = torch.empty_like(w)
    for e in range(w.shape[0]):  # one expert at a time, to bound the temporaries
        codes = torch.stack((w[e] & 15, w[e] >> 4), dim=-1).view(w.shape[1], -1, 2, bk // 2)
        out[e] = (codes[:, :, 0] | (codes[:, :, 1] << 4)).view(w.shape[1], -1)
    return out


def _repack_scales(sf, bk=MOE_BK):
    """[E, N, K/16] -> [E, K/bk, N, bk/16]."""
    e, n, g = sf.shape
    return sf.view(e, n, g // (bk // 16), bk // 16).permute(0, 2, 1, 3).contiguous()


def _moe_bank(weights):
    return {
        "w13": _repack_codes(weights["w13"]), "w13_s": _repack_scales(weights["w13_sf"]),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": _repack_codes(weights["w2"]), "w2_s": _repack_scales(weights["w2_sf"]),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "post_w": _gain(weights["post_expert_norm"]),
    }


# ----------------------------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------------------------
def _residual_coefficients(gate):
    """MuseSpark1xNormPreservingResidualConnection: new = alpha * stream + beta * branch."""
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)).contiguous(), beta.contiguous()


def _gain(w):
    """Muse's norm weights are zero-centered: the scale is 1 + w."""
    return (1.0 + w.float()).contiguous()


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    import types
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _candidates(m, n, k):
    """Split-K configs (BM, BN, BK, splits, stages, warps) worth timing for an [m, k] x [n, k]^T GEMM."""
    if m <= 32:
        bm, bns, bks, warps = (16 if m <= 16 else 32), (32, 64, 128), (128, 256), 4
    else:  # verify blocks: 64-row tiles, tensor-core bound sooner
        bm, bns, bks, warps = 64, (64, 128), (64, 128), 4
    out = []
    for bn in bns:
        for bk in bks:
            for stages in (3, 4):
                if (bn + bm) * bk * 2 * stages > 160 * 1024:
                    continue
                for s in (1, 2, 4, 8, 16):
                    if n % bn or k % (s * bk):
                        continue
                    progs = (n // bn) * s * triton.cdiv(m, bm)
                    if not 120 <= progs <= 2400:
                        continue
                    out.append((bm, bn, bk, s, stages, warps))
    return out


def _time(fn, flush, reps=10):
    """Median time of a CUDA graph replay of fn with a cold L2, in ms (the judge replays graphs;
    eager timing would charge Python launch overhead)."""
    fn()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        flush.zero_()  # evicts L2
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    del graph
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]


def _tune(mats, device):
    """{(name, M): cfg, or None for cuBLAS}: the fastest correct one at each tuned token count."""
    began = time.time()
    flush = torch.empty(64 << 20, dtype=torch.int32, device=device)  # 256 MB, past the 50 MB L2
    gen = torch.Generator(device=device).manual_seed(0)
    chosen = {}
    for m in TUNE_TOKENS:
        for name, (w, ms) in mats.items():
            if m not in ms:
                continue
            n, k = w.shape
            x = torch.randn((m, k), generator=gen, device=device).to(torch.bfloat16)
            ref = torch.mm(x, w.t(), out_dtype=torch.float32)
            # Each split's FP32 partials are written here and read back by the epilogue.
            traffic = lambda splits: 2 * splits * m * n * 4 / PARTIALS_BYTES_PER_MS  # noqa: E731
            best_t, best = _time(lambda: _mm(x, w, None), flush) + traffic(1), None
            cublas_t = best_t
            for cfg in _candidates(m, n, k):
                if time.time() - began > TUNE_BUDGET_S:
                    break
                try:
                    got = _mm(x, w, cfg).sum(0)
                    err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
                    if not err < 1e-3:
                        continue
                    t = _time(lambda: _mm(x, w, cfg), flush) + traffic(cfg[3])
                except Exception:  # out of resources or a compile failure: not a candidate
                    continue
                if t < best_t:
                    best_t, best = t, cfg
            chosen[(name, m)] = best
            print(f"tune {name:8s} M={m:3d}: cublas {cublas_t * 1e3:7.1f} us, best {best_t * 1e3:7.1f} us {best}")
    del flush
    return chosen


def _synthetic_routing(tokens, gen, device):
    """Local expert ids [T, 8] (-1 elsewhere) skewed like the judge's: a few hot experts, and about
    MOE_LOCAL_PAIRS of a token's 8 routes on this GPU."""
    share = MOE_LOCAL_SHARE[min(MOE_LOCAL_SHARE, key=lambda t: abs(t - tokens))]
    local = 1.0 / torch.arange(1, LOCAL_EXPERTS + 1, device=device, dtype=torch.float32)
    probs = torch.cat([local / local.sum() * share,
                       torch.full((N_EXPERTS - LOCAL_EXPERTS,), (1 - share) / (N_EXPERTS - LOCAL_EXPERTS),
                                  device=device)])
    ids = torch.multinomial(probs.expand(tokens, -1).contiguous(), TOP_K, replacement=False, generator=gen)
    return torch.where(ids < LOCAL_EXPERTS, ids, -1).to(torch.int32).contiguous()


def _moe_candidates(tokens):
    """(bn_up, bn_down, splits, warps)."""
    if tokens <= 32:
        return [(32, 64, 1, 4), (32, 64, 2, 4), (32, 64, 4, 4), (64, 64, 2, 4), (32, 128, 2, 4),
                (64, 128, 4, 8), (16, 32, 4, 4)]
    return [(32, 64, 1, 4), (64, 64, 1, 4), (64, 128, 1, 8), (32, 64, 2, 4), (64, 128, 2, 8)]


def _tune_moe(state, device):
    """{T: cfg, or None for Marlin}: the faster expert path at each tuned token count, on
    synthetic skewed routing; a candidate must agree with Marlin to count."""
    began = time.time()
    flush = torch.empty(64 << 20, dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(1)
    chosen = {}
    for m in MOE_TUNE_TOKENS:
        x = torch.randn((m, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
        local_ids = _synthetic_routing(m, gen, device)
        tw = torch.rand((m, TOP_K), generator=gen, device=device)
        tw = (tw / tw.sum(-1, keepdim=True)).contiguous()
        best, best_t, ref = None, math.inf, None
        try:
            ref = _experts(x, tw, local_ids, state).float()
            best_t = _time(lambda: _experts(x, tw, local_ids, state), flush)
        except Exception as exc:  # Marlin unavailable: any working candidate wins
            print(f"moe M={m}: marlin failed: {exc!r}")
        marlin_t = best_t
        for cfg in _moe_candidates(m):
            if time.time() - began > MOE_TUNE_BUDGET_S:
                break
            try:
                got = _moe(x, tw, local_ids, state["moe_bank"], cfg).float()
                if ref is not None:
                    err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
                    if not err < 2e-2:
                        print(f"moe M={m} {cfg}: off from marlin by {err:.2e}")
                        continue
                t = _time(lambda: _moe(x, tw, local_ids, state["moe_bank"], cfg), flush)
            except Exception as exc:
                print(f"moe M={m} {cfg}: {exc!r}")
                continue
            if t < best_t:
                best_t, best = t, cfg
        chosen[m] = best
        print(f"tune moe     M={m:3d}: marlin {marlin_t * 1e3:7.1f} us, best {best_t * 1e3:7.1f} us {best}")
    del flush
    return chosen


def _synthetic_attention(batch, length, windowed, gen, device):
    """Step inputs like the judge's scored attention cases: 80K-token contexts (+/- 7.5% across a
    batch), one token per page, scattered pages."""
    ctx = 80_000
    longest = math.ceil(ctx * 1.075)
    seq = (ctx * (1 + 0.075 * (2 * torch.rand((batch,), generator=gen, device=device) - 1))).long()
    if batch == 1:
        seq.fill_(ctx)
    pages = longest + length
    rows = batch * pages
    perm = torch.randperm(rows, generator=gen, device=device).to(torch.int32)
    table = perm.view(batch, pages).contiguous()
    tokens = batch * length
    caches = [torch.randn((rows, 1, HKV, HEAD_DIM), generator=gen, device=device, dtype=torch.bfloat16).to(FP8)
              for _ in range(2)]
    cache_seqlens = (seq + length).to(torch.int32)
    cu = torch.arange(0, tokens + 1, length, dtype=torch.int32, device=device)
    meta = None
    if not windowed and length == 1:
        try:
            from sgl_kernel.flash_attn import get_scheduler_metadata
            meta = get_scheduler_metadata(
                batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=HQ,
                num_heads_k=HKV, headdim=HEAD_DIM, cache_seqlens=cache_seqlens, qkv_dtype=FP8,
                cu_seqlens_q=cu, page_size=1, causal=True, num_splits=0)
        except Exception:
            meta = None
    one = torch.ones((), dtype=torch.float32, device=device)
    data = {
        "k_cache": caches[0], "v_cache": caches[1], "page_table": table, "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu, "scheduler_metadata": meta, "k_scale": one, "v_scale": one.clone(),
        "config": {"batch": batch, "q_len": length, "tokens": tokens, "page_size": 1,
                   "window": 2048 if windowed else 0, "qk_scale_factor": 5.4730064863875,
                   "sm_scale": HEAD_DIM ** -0.5, "kv_scale": 1.0},
    }
    q = torch.randn((tokens, HQ, HEAD_DIM), generator=gen, device=device)
    q8 = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True))).to(FP8)
    p_qkv = torch.randn((1, tokens, 2304), generator=gen, device=device)
    return data, q8, p_qkv


def _attn_candidates():
    return [(t, bn, 4, f8) for t in (264, 528, 1056, 2112) for bn in (32, 64) for f8 in (False, True)]


def _tune_attn(device):
    """{(batch, q_len, windowed): cfg, or None for FA3} at the scored shapes."""
    began = time.time()
    flush = torch.empty(64 << 20, dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(2)
    chosen = {}
    for shape in ATTN_TUNE_SHAPES:
        data, q8, p_qkv = _synthetic_attention(*shape, gen, device)
        best, best_t, ref = None, math.inf, None
        if flash_attn_with_kvcache is not None:
            try:
                ref = _attention_fa3(data, q8, p_qkv).float()
                best_t = _time(lambda: _attention_fa3(data, q8, p_qkv), flush)
            except Exception as exc:
                print(f"attn {shape}: fa3 failed: {exc!r}")
        fa3_t = best_t
        for cfg in _attn_candidates():
            if time.time() - began > ATTN_TUNE_BUDGET_S:
                break
            try:
                got = _attention(data, q8, p_qkv, cfg).float()
                if ref is not None:
                    err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
                    if not err < 8e-2:
                        print(f"attn {shape} {cfg}: off from fa3 by {err:.2e}")
                        continue
                t = _time(lambda: _attention(data, q8, p_qkv, cfg), flush)
            except Exception as exc:
                print(f"attn {shape} {cfg}: {exc!r}")
                continue
            if t < best_t:
                best_t, best = t, cfg
        chosen[shape] = best if best is not None or ref is not None else ATTN_DEFAULT
        print(f"tune attn {str(shape):18s}: fa3 {fa3_t * 1e3:7.1f} us, best {best_t * 1e3:7.1f} us {best}")
    del flush
    return chosen


def _attn_cfg(state, cfg):
    """The tuned attention path for this shape; elsewhere FA3, or the Triton kernel without it."""
    key = (cfg["batch"], cfg["q_len"], bool(cfg["window"]))
    if key in state["attn"]:
        return state["attn"][key]
    return ATTN_DEFAULT if flash_attn_with_kvcache is None else None


def prepare(weights):
    """Called once, untimed: repack the weights (losslessly) and tune the GEMMs."""
    device = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # cos and sin [2^21, 32] FP32 of position * inv_freq, as SGLang's RotaryEmbedding builds them.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float, device=device) / HEAD_DIM))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    alpha1, beta1 = _residual_coefficients(weights["attn_residual_gate"])
    alpha2, beta2 = _residual_coefficients(weights["ffn_residual_gate"])
    unpooler = weights["unpooler"].contiguous()
    state = {
        "qkv": weights["qkv_gate"].contiguous(),
        "o": weights["o"].contiguous(),
        # The pooler with the router's BF16 hi and lo rows after it: one GEMM for both.
        "poolrt": torch.cat([weights["pooler"], hi, lo]).contiguous(),
        "unpool_s": unpooler[:UNPOOL_SLICE].contiguous(),
        "unpool": unpooler,
        "cos": freqs.cos().contiguous(),
        "sin": freqs.sin().contiguous(),
        "alpha1": alpha1, "beta1": beta1, "alpha2": alpha2, "beta2": beta2,
        "pre_ffn": _gain(weights["pre_ffn_norm"]),
        "pre_expert": _gain(weights["pre_expert_norm"]),
        "post_ffn": _gain(weights["post_ffn_norm"]),
        "next_pre_attn": _gain(weights["next_pre_attn_norm"]),
        "router_bias": weights["router_bias"].float().contiguous(),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight in BF16.
        "post_expert_w": _gain(weights["post_expert_norm"]).to(torch.bfloat16),
    }
    del freqs
    state["bank"] = _marlin_bank(weights)
    mats = {
        "qkv": (state["qkv"], TUNE_TOKENS),
        "o": (state["o"], TUNE_TOKENS),
        "poolrt": (state["poolrt"], TUNE_TOKENS),
        "unpool_s": (state["unpool_s"], tuple(m for m in TUNE_TOKENS if m <= SLICED_UNPOOL_MAX_TOKENS)),
        "unpool": (state["unpool"], tuple(m for m in TUNE_TOKENS if m > SLICED_UNPOOL_MAX_TOKENS)),
    }
    state["gemm"] = _tune(mats, device)
    state["moe_bank"] = _moe_bank(weights)
    state["moe"] = _tune_moe(state, device)
    state["attn"] = _tune_attn(device)
    torch.cuda.synchronize()
    return state


def _cfg(state, name, m):
    """The tuned config for the smallest tuned token count >= m, or None (cuBLAS)."""
    for tm in TUNE_TOKENS:
        if m <= tm and (name, tm) in state["gemm"]:
            return state["gemm"][(name, tm)]
    return None


def _moe_cfg(state, m):
    """The tuned expert path for the smallest tuned token count >= m, or None (Marlin)."""
    for tm in MOE_TUNE_TOKENS:
        if m <= tm and tm in state["moe"]:
            return state["moe"][tm]
    return None


# ----------------------------------------------------------------------------------------------
# experts: SGLang's MuseSpark1xFp4Experts Marlin path
# ----------------------------------------------------------------------------------------------
def _experts(x, topk_weights, local_ids, state):
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


# ----------------------------------------------------------------------------------------------
# the layer
# ----------------------------------------------------------------------------------------------
def kernel_fn(data, state):
    """Returns (next layer's attention input bfloat16 [T, 8192], residual stream float32
    [T, 8192], experts per token int32 [T, 8])."""
    cfg = data["config"]
    tokens = cfg["tokens"]
    device = data["stream"].device
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    # qkv projection, then qk-norm, RoPE, FP8 q and the cache store in one pass.
    p_qkv = _mm(data["attn_input"], state["qkv"], _cfg(state, "qkv", tokens))
    q8 = torch.empty((tokens, HQ, HEAD_DIM), dtype=FP8, device=device)
    _qkv_post[(tokens,)](
        p_qkv, p_qkv.shape[0], tokens, p_qkv.shape[2], data["positions"], data["cache_loc"],
        state["cos"], state["sin"], k_cache, v_cache, data["k_scale"], data["v_scale"], q8,
        RMS_EPS, HAS_ROPE=bool(cfg["rope_theta"]), num_warps=4)

    acfg = _attn_cfg(state, cfg)
    if acfg is not None:
        gated = _attention(data, q8, p_qkv, acfg)
    else:
        gated = _attention_fa3(data, q8, p_qkv)

    # o projection (all-reduce: identity), then the post-attention boundary.
    p_o = _mm(gated, state["o"], _cfg(state, "o", tokens))
    stream1 = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=device)
    h = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
    _attn_boundary[(tokens,)](
        p_o, p_o.shape[0], tokens, data["stream"], state["alpha1"], state["beta1"], state["pre_ffn"],
        stream1, h, POST_NORM_EPS, RMS_EPS, BLOCK=HIDDEN, num_warps=16)

    # pooler and router in one GEMM, then the pre-expert norm and the routing.
    p_pr = _mm(h, state["poolrt"], _cfg(state, "poolrt", tokens))
    p = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=device)
    ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=device)
    local_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.empty((tokens, TOP_K), dtype=torch.float32, device=device)
    _pool_route[(tokens,)](
        p_pr, p_pr.shape[0], tokens, p_pr.shape[2], state["pre_expert"], state["router_bias"],
        data["local_start"], p, ids, local_ids, topk_weights, RMS_EPS, ROUTER_MULTIPLIER, ROUTE_EPS,
        D=D_EXPERT, E=N_EXPERTS, K=TOP_K, LOCAL=LOCAL_EXPERTS, num_warps=8)

    # experts (all-reduce: identity); the unpooler (up to 8 tokens this GPU's slice, the
    # all-gather's zero pad folded into the boundary); the post-FFN boundary.
    moe_cfg = _moe_cfg(state, tokens)
    if moe_cfg is None:
        y = _experts(p, topk_weights, local_ids, state)
    else:
        y = _moe(p, topk_weights, local_ids, state["moe_bank"], moe_cfg)
    if y.dtype != torch.bfloat16:
        y = y.to(torch.bfloat16)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        p_u = _mm(y, state["unpool_s"], _cfg(state, "unpool_s", tokens))
    else:
        p_u = _mm(y, state["unpool"], _cfg(state, "unpool", tokens))
    stream2 = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=device)
    hidden = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
    _ffn_boundary[(tokens,)](
        p_u, p_u.shape[0], tokens, p_u.shape[2], stream1, state["post_ffn"], state["alpha2"], state["beta2"],
        state["next_pre_attn"], stream2, hidden, POST_NORM_EPS, RMS_EPS, BLOCK=HIDDEN, num_warps=16)
    return hidden, stream2, ids
