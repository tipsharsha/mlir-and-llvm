# Open Kernel Challenge: `muse_decode_layer`

One whole decoder layer of Muse Spark Open, as one H200 runs it under TP8/EP8. The submission is timed against SGLang's version of the layer, which takes 22.62 ms per decode step.

## Where the time goes

The layer's parts are also scored as separate problems. Their boards show what each part costs SGLang per decode step, and what others have reached on it:

| Part | SGLang | Board rank 1 | Board rank 50 |
|---|---|---|---|
| Routed experts (NVFP4 MoE, Marlin) | 13.45 ms | 2.58× | 2.20× |
| Attention (FA3, FP8 cache, page size 1) | 3.77 ms | 3.03× | 2.37× |
| Dense projections (cuBLAS) | 4.83 ms | 1.64× | 1.52× |
| Glue (norms, routing, cache writes) | ~0.6 ms | | |

Rank 50 on this problem is 1.96× (about 11.5 ms). Reaching it means about rank-50 quality on all three parts, and the experts are about two-thirds of the needed saving.

## What `submission.py` does

Every SGLang piece that has a replacement stays as a fallback. `prepare()` times each one against its replacement on the scored shapes, using CUDA-graph replays with a cold L2 cache, and keeps the faster. It checks each candidate's output against SGLang's first, so a broken candidate is never chosen.

| Step | Replacement | Fallback |
|---|---|---|
| qkv_gate, o, pooler + router, unpooler | split-K Triton GEMMs writing FP32 partial sums (T ≤ 32) | cuBLAS |
| QK-norm, RoPE, FP8 q, FP8 k/v cache store | `_qkv_post`: 1 kernel reading the partials | (always used) |
| Attention | `_attn_chunk`: split-KV flash-decoding. Each program takes one request × one key chunk, for both KV heads, so each page-1 row gather reads one 128-byte line. Optional FP8 q·k. | FA3 |
| Chunk merge + per-head norm × sigmoid(gate) | `_attn_combine`: 1 kernel per (token, head) | FA3 + `_norm_gate` |
| Post-attention norm, residual gate, pre-FFN norm | `_attn_boundary` | (always used) |
| Router | its BF16 hi/lo rows appended under the pooler: one GEMM | (always used) |
| Pre-expert norm, sigmoid top-8, renormalize, local ids | `_pool_route` | (always used) |
| Experts | `_moe_plan` (expert-sorted pairs + work blocks) → `_moe_up` (gate/up + SiLU fused) → `_moe_down` (split-K) → `_moe_final` (per-expert norm, weighted sum) | Marlin |
| Unpooler zero pad, post-FFN norm, residual, next norm | `_ffn_boundary` | (always used) |

**The NVFP4 expert kernels.** In `prepare()` the weights are repacked bit for bit. Within each 256-element K chunk, byte j holds element j and element j+128, so the low and high nibbles multiply contiguous halves of the activations. The block scales are regrouped per chunk, so each tile's scales are contiguous.

Decoding works like this:
- An E2M1 code becomes an FP16 bit pattern: `(code & 7) << 9 | sign << 15`.
- It is multiplied by its E4M3 block scale. The product is exact in BF16.
- The 2⁻¹⁴ the decoding leaves, and the FP32 tensor scale, are applied to the accumulator.

So the only rounding is the BF16 activations and the FP32 accumulation.

Precision follows the rules:
- BF16 weights stay BF16, and NVFP4 weights stay NVFP4, repacked only.
- The router runs as the allowed BF16 hi+lo pair with FP32 accumulation.
- Queries are FP8 E4M3, as FA3's.
- The cache stays FP8.
- Every key in the window or context is attended to.

## Testing without a GPU

`tests/` runs every Triton kernel in the CPU interpreter (`TRITON_INTERPRET=1`). `fp8fix.py` patches two interpreter bugs:
- A float→e4m3 cast that halves values rounding into the next exponent.
- BF16 `tl.dot` multiplying the raw 16-bit patterns as integers.

| Test | Checks |
|---|---|
| `test_kernels.py` | each epilogue kernel against PyTorch |
| `test_gemm.py` | split-K GEMM |
| `test_moe.py` | Triton MoE against the reference's FP32 expert math (rel. error ~5e-7) |
| `test_attention.py` | Triton attention against an FP32 FA3 stand-in (`fakes.py`): sliding/full, decode/verify, page sizes 1 and 64, KV scale 1/64, both q·k precisions |
| `test_e2e.py` | the whole `kernel_fn`, with the Triton GEMMs, attention and MoE forced on, through the judge's own `check_implementation` (`E2E_FA3=1` uses the FA3 stand-in instead) |

Setup:

```bash
okc init muse_decode_layer /tmp/okc        # reference.py, nvfp4.py, checks.py
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install triton numpy
cd tests && OKC_DIR=/tmp/okc/muse_decode_layer python test_e2e.py
```

The interpreter re-reads kernel source from disk, so don't edit `submission.py` while a test runs. Run the tests on a copy if you need to keep editing.

## Submitting

The rules have to be accepted by a person (`okc login`). Then:

```bash
okc check muse_decode_layer submission.py    # off the board, 20 a day
okc submit muse_decode_layer submission.py   # 10 a day
```

Not yet run on the judge, so there are no timings yet. `prepare()` prints a tuning table (SGLang's time against the best replacement, per GEMM, expert and attention shape), which comes back in the run's log.
