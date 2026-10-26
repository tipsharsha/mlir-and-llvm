"""The Triton split-KV attention (and its fused combine + gated norm) against FA3's stand-in."""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("OKC_DIR", ".")]  # this submission, then okc init's reference.py and nvfp4.py
import torch
import fp8fix  # noqa: F401
import submission as S
import reference as R
from fakes import fake_fa3

S.flash_attn_with_kvcache = fake_fa3
gen = torch.Generator().manual_seed(0)
cases = [dict(batch=1, ctx=10, layer="sliding"),
         dict(batch=2, ctx=2500, layer="sliding", page=1),
         dict(batch=3, ctx=300, layer="full", q_len=8),
         dict(batch=2, ctx=3000, layer="sliding", q_len=8, kv_scale=0.015625),
         dict(batch=4, ctx=700, layer="full", page=1, spread="mixed")]
for i, c in enumerate(cases):
    data = R.generate_input(**{**c, "seed": 200 + i, "device": "cpu"})
    t = data["config"]["tokens"]
    q = torch.randn((t, 16, 64), generator=gen)
    q8 = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True))).to(S.FP8)
    p_qkv = torch.randn((2, t, 2304), generator=gen)
    ref = S._attention_fa3(data, q8, p_qkv).float()
    for acfg in ((1024, 64, 4, False), (64, 32, 4, True)):
        got = S._attention(data, q8, p_qkv, acfg).float()
        chunk, nc = S._attn_geometry(data["config"], data["page_table"].shape[1] * data["config"]["page_size"], acfg[0])
        print(c, acfg, f"chunks {nc}x{chunk}", f"rel {float((got - ref).norm() / ref.norm()):.2e}")
