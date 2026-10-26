"""The Triton MoE path against the reference's expert math, in the CPU interpreter (FP32
activations: the interpreter's BF16 dot is broken)."""
import os, sys, time
os.environ["TRITON_INTERPRET"] = "1"
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("OKC_DIR", ".")]  # this submission, then okc init's reference.py and nvfp4.py
import torch
import fp8fix  # noqa: F401
import submission as S
from nvfp4 import dequantize, random_weights

gen = torch.Generator().manual_seed(0)
E, D = 32, 4096
w13, w13_sf, w13_gs = random_weights(2 * D, D, generator=gen, device="cpu", leading=(E,))
w2, w2_sf, w2_gs = random_weights(D, D, generator=gen, device="cpu", leading=(E,))
w13_gs = w13_gs.expand(E).clone() * (1 + 0.1 * torch.rand(E, generator=gen))
w2_gs = w2_gs.expand(E).clone() * (1 + 0.1 * torch.rand(E, generator=gen))
post = (torch.randn(D, generator=gen) * 0.1).bfloat16()
W = dict(w13=w13, w13_sf=w13_sf, w13_gs=w13_gs, w2=w2, w2_sf=w2_sf, w2_gs=w2_gs, post_expert_norm=post)
bank = S._moe_bank(W)

def reference(x, ids, tw):
    out = torch.zeros(x.shape)
    for e in torch.unique(ids).tolist():
        if not 0 <= e < E: continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequantize(w13[e], w13_sf[e], w13_gs[e]).T
        inter = torch.nn.functional.silu(h[:, :D]) * h[:, D:]
        y = (inter @ dequantize(w2[e], w2_sf[e], w2_gs[e]).T) * (1 + post.float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * tw[tok, slot, None])
    return out

for T, cfg in ((1, (32, 64, 4, 4)), (3, (32, 64, 2, 4)), (20, (64, 128, 1, 4))):
    x = torch.randn(T, D, generator=gen)
    # Skewed routing: a few local experts are popular; the rest of each top-8 is elsewhere.
    ids = torch.stack([torch.randperm(256, generator=gen)[:8] for _ in range(T)])
    hot = torch.tensor([3, 7, 7, 30])
    ids[:, 0] = hot[torch.randint(0, 4, (T,), generator=gen)]
    ids[:, 1] = torch.where(ids[:, 1] == ids[:, 0], (ids[:, 0] + 100) % 256, ids[:, 1])
    ids = torch.stack([torch.unique(r)[:0].new_tensor(r.tolist()) for r in ids])  # distinct per row already
    tw = torch.rand(T, 8, generator=gen); tw /= tw.sum(-1, keepdim=True)
    local = torch.where(ids < 32, ids, -1).to(torch.int32).contiguous()
    t0 = time.time()
    got = S._moe(x, tw, local, bank, cfg)
    ref = reference(x, local.long(), tw)
    print(f"T={T} cfg={cfg} pairs={int((local >= 0).sum())} rel={float((got - ref).norm() / ref.norm()):.2e} "
          f"rows-zero-ok={bool((got[(local < 0).all(-1)] == 0).all())} {time.time() - t0:.0f}s")
