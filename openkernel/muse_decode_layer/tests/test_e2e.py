import os, sys; os.environ["TRITON_INTERPRET"]="1"
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("OKC_DIR", ".")]  # this submission, then okc init's reference.py and nvfp4.py
import torch
import fp8fix  # noqa: F401  (patches the interpreter)
torch.cuda.synchronize = lambda *a, **k: None
import submission as S, reference as R
_mm0 = torch.mm
def _mm_shim(a, b, *, out_dtype=None):
    return a.float() @ b.float() if out_dtype is not None else _mm0(a, b)
torch.mm = _mm_shim
from nvfp4 import dequantize

from fakes import fake_fa3

W = None
def fake_experts(x, tw, lid, state):
    out = torch.zeros(x.shape)
    for e in torch.unique(lid).tolist():
        if e < 0: continue
        tok, slot = (lid == e).nonzero(as_tuple=True)
        w13 = dequantize(W["w13"][e], W["w13_sf"][e], W["w13_gs"][e]).bfloat16()
        h = (x[tok] @ w13.T)
        inter = (torch.nn.functional.silu(h[:, :4096].float()) * h[:, 4096:].float()).bfloat16()
        w2 = dequantize(W["w2"][e], W["w2_sf"][e], W["w2_gs"][e]).bfloat16()
        y = (inter @ w2.T).float() * state["post_expert_w"].float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * tw[tok, slot, None])
    return out.bfloat16()

# The Triton attention by default; E2E_FA3=1 runs FA3's stand-in instead.
S.flash_attn_with_kvcache = fake_fa3 if os.environ.get("E2E_FA3") else None
S._tune_attn = lambda device: {}
S._experts = fake_experts
S._marlin_bank = lambda w: None
# Force the Triton paths (the GPU picks by timing): the first split-K candidate per GEMM, and the
# Triton experts at every token count.
S._tune = lambda mats, device: {(name, m): S._candidates(m, *w.shape)[0]
                                for name, (w, ms) in mats.items() for m in ms}
S._tune_moe = lambda state, device: {m: (32, 64, 2, 4) for m in S.MOE_TUNE_TOKENS}
W = R.generate_weights(5353, device="cpu")
state = S.prepare(W)
cases = [dict(batch=1, ctx=10, layer="sliding"),
         dict(batch=3, ctx=300, layer="full", q_len=8, dist="outliers"),
         dict(batch=4, ctx=700, layer="sliding", q_len=8, page=1, kv_scale=0.015625),
         dict(batch=9, ctx=500, layer="sliding", spread="mixed", page=1),
         dict(batch=2, ctx=3000, layer="sliding", q_len=1),
         dict(batch=5, ctx=400, layer="sliding", q_len=8, spread="mixed", page=1)]
for i, c in enumerate(cases):
    c = {**c, "seed": 100 + i, "device": "cpu"}
    ls = R.case_fields(c, W)["local_start"]
    data = R.generate_input(**{**c, "local_start": ls})
    out = S.kernel_fn(data, state)
    raw = R.collect(data, out)
    print(c, R.check_implementation(data, raw, W))
