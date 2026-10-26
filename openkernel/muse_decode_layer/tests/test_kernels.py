import os, sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.environ.get("OKC_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import fp8fix
import submission as S

torch.manual_seed(0)
def rel(a, b): return float((a.float() - b.float()).norm() / b.float().norm())
def rms(x, eps): x = x.float(); return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

# GEMM
for (m, n, k, cfg) in [(1, 64, 512, (16, 32, 128, 2, 3, 4)), (5, 128, 1024, (16, 64, 256, 4, 3, 4)),
                       (20, 64, 512, (32, 32, 128, 1, 3, 4))]:
    x = torch.randn(m, k).bfloat16(); w = torch.randn(n, k).bfloat16()
    got = S._mm(x, w, cfg)
    print("gemm", m, n, k, got.shape, rel(got.sum(0), x.float() @ w.float().T))

# qkv_post
T, Sn = 3, 2
part = torch.randn(Sn, T, 2304)
full = part.sum(0)
pos = torch.tensor([5, 1000, 70000])
loc = torch.tensor([7, 2, 11])
inv_freq = 1.0 / (S.ROPE_THETA ** (torch.arange(0, 64, 2).float() / 64))
freqs = torch.arange(80000).float()[:, None] * inv_freq
cos, sin = freqs.cos().contiguous(), freqs.sin().contiguous()
for rope in (True, False):
    for scale in (1.0, 1 / 64):
        kc = torch.zeros(16, 1, 2, 64, dtype=S.FP8); vc = torch.zeros(16, 1, 2, 64, dtype=S.FP8)
        q8 = torch.empty(T, 16, 64, dtype=S.FP8)
        ks = torch.tensor(scale); vs = torch.tensor(scale)
        S._qkv_post[(T,)](part, Sn, T, 2304, pos, loc, cos, sin, kc, vc, ks, vs, q8, 1e-5, HAS_ROPE=rope)
        q = rms(full[:, :1024].view(T, 16, 64), 1e-5); k = rms(full[:, 1024:1152].view(T, 2, 64), 1e-5)
        v = full[:, 1152:1280].view(T, 2, 64)
        if rope:
            def rp(x):
                ang = pos.float()[:, None, None] * inv_freq
                e, o = x[..., 0::2], x[..., 1::2]
                return torch.stack([e * ang.cos() - o * ang.sin(), e * ang.sin() + o * ang.cos()], -1).flatten(-2)
            q, k = rp(q), rp(k)
        print("qkv_post rope", rope, "scale", scale, "q", rel(q8.float(), q.to(S.FP8).float()),
              "k", rel(kc.view(-1, 2, 64)[loc].float(), (k / scale).to(S.FP8).float()),
              "v", rel(vc.view(-1, 2, 64)[loc].float(), (v / scale).to(S.FP8).float()),
              "untouched", int(kc.view(-1,128)[0].float().abs().sum()))

# norm_gate
attn = torch.randn(T, 1024).bfloat16()
out = torch.empty(T, 1024, dtype=torch.bfloat16)
S._norm_gate[(T,)](attn, part, Sn, T, 2304, out, 1e-5)
ref = (rms(attn.view(T, 16, 64), 1e-5).reshape(T, 1024) * torch.sigmoid(full[:, 1280:]))
print("norm_gate", rel(out, ref))

# attn boundary
p_o = torch.randn(2, T, 8192)
stream = torch.randn(T, 8192)
gate = torch.randn(8192).bfloat16() * 0.3
a1, b1 = S._residual_coefficients(gate)
wpre = S._gain(torch.randn(8192).bfloat16() * 0.1)
s1 = torch.empty(T, 8192); h = torch.empty(T, 8192, dtype=torch.bfloat16)
S._attn_boundary[(T,)](p_o, 2, T, stream, a1, b1, wpre, s1, h, 1e-8, 1e-5, BLOCK=8192)
rs = a1 * stream + b1 * rms(p_o.sum(0), 1e-8)
print("attn_boundary", rel(s1, rs), rel(h, rms(rs, 1e-5) * wpre))

# pool_route
N = 4608
pp = torch.randn(2, T, N) * 0.5
wpe = S._gain(torch.randn(4096).bfloat16() * 0.1)
bias = torch.randn(256) * 0.05
start = torch.tensor([32], dtype=torch.int32)
p = torch.empty(T, 4096, dtype=torch.bfloat16)
ids = torch.empty(T, 8, dtype=torch.int32); lid = torch.empty_like(ids); tw = torch.empty(T, 8)
S._pool_route[(T,)](pp, 2, T, N, wpe, bias, start, p, ids, lid, tw, 1e-5, S.ROUTER_MULTIPLIER, 1e-15,
                    D=4096, E=256, K=8, LOCAL=32)
fp = pp.sum(0)
print("pool", rel(p, rms(fp[:, :4096], 1e-5) * wpe))
logits = fp[:, 4096:4352] + fp[:, 4352:]
sc = torch.sigmoid(logits * S.ROUTER_MULTIPLIER)
rid = (sc + bias).topk(8, -1).indices
print("ids match", bool((ids.long().sort(-1).values == rid.sort(-1).values).all()))
pk = sc.gather(1, ids.long())
print("tw", rel(tw, pk / pk.sum(-1, keepdim=True)))
loc_ref = (ids.long() - 32) % 256
loc_ref = torch.where(loc_ref < 32, loc_ref, -1)
print("local", bool((lid.long() == loc_ref).all()))

# ffn boundary, sliced (NF=1024) and full
for NF in (1024, 8192):
    pu = torch.randn(3, T, NF)
    wf = S._gain(torch.randn(8192).bfloat16() * 0.1); wn = S._gain(torch.randn(8192).bfloat16() * 0.1)
    a2, b2 = S._residual_coefficients(torch.randn(8192).bfloat16() * 0.3)
    s2 = torch.empty(T, 8192); hid = torch.empty(T, 8192, dtype=torch.bfloat16)
    S._ffn_boundary[(T,)](pu, 3, T, NF, s1, wf, a2, b2, wn, s2, hid, 1e-8, 1e-5, BLOCK=8192)
    f = torch.zeros(T, 8192); f[:, :NF] = pu.sum(0)
    rs2 = a2 * s1 + b2 * rms(f * wf, 1e-8)
    print("ffn_boundary", NF, rel(s2, rs2), rel(hid, rms(rs2, 1e-5) * wn))
