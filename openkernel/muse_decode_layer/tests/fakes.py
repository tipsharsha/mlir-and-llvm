"""Test-only stand-ins for SGLang's FA3: plain PyTorch attention in FP32."""
import torch


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
