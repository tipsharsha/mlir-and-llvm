import os, sys; os.environ["TRITON_INTERPRET"]="1"
sys.path.insert(0, os.environ.get("OKC_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import fp8fix
torch.cuda.synchronize = lambda *a, **k: None
import submission as S, reference as R
_mm0 = torch.mm
def _mm_shim(a, b, *, out_dtype=None):
    return a.float() @ b.float() if out_dtype is not None else _mm0(a, b)
torch.mm = _mm_shim
from nvfp4 import dequantize

def fake_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale,
             causal, window_size, scheduler_metadata=None, k_descale=None, v_descale=None):
    T = q.shape[0]; L = max_seqlen_q; B = T // L
    page = k_cache.shape[1]
    ks = 1.0 if k_descale is None else float(k_descale[0, 0]); vs = 1.0 if v_descale is None else float(v_descale[0, 0])
    kf = k_cache.view(-1, 2, 64); vf = v_cache.view(-1, 2, 64)
    out = torch.empty(T, 16, 64)
    for b in range(B):
        n = int(cache_seqlens[b]); t = torch.arange(n)
        rows = page_table[b].long()[t // page] * page + t % page
        K = kf[rows].float() * ks; V = vf[rows].float() * vs
        for i in range(L):
            qp = n - L + i
            vis = t <= qp
            if window_size[0] >= 0: vis &= t >= qp - window_size[0]
            qq = q[b * L + i].float().view(2, 8, 64)
            att = torch.einsum("kgd,skd->kgs", qq, K) * softmax_scale
            att = att.masked_fill(~vis[None, None], float("-inf")).softmax(-1)
            out[b * L + i] = torch.einsum("kgs,skd->kgd", att, V).reshape(16, 64)
    return out.bfloat16()

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

S.flash_attn_with_kvcache = fake_fa3
S._experts = fake_experts
S._marlin_bank = lambda w: None
S._tune = lambda mats, device: {}
W = R.generate_weights(5353, device="cpu")
state = S.prepare(W)
cases = [dict(batch=1, ctx=10, layer="sliding"),
         dict(batch=3, ctx=300, layer="full", q_len=8, dist="outliers"),
         dict(batch=4, ctx=700, layer="sliding", q_len=8, page=1, kv_scale=0.015625),
         dict(batch=9, ctx=500, layer="sliding", spread="mixed", page=1),
         dict(batch=2, ctx=3000, layer="sliding", q_len=1)]
for i, c in enumerate(cases):
    c = {**c, "seed": 100 + i, "device": "cpu"}
    ls = R.case_fields(c, W)["local_start"]
    data = R.generate_input(**{**c, "local_start": ls})
    out = S.kernel_fn(data, state)
    raw = R.collect(data, out)
    print(c, R.check_implementation(data, raw, W))
