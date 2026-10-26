# Open Kernel Challenge: `muse_decode_layer`

One whole decoder layer of Muse Spark Open, as one H200 runs it under TP8/EP8. The submission is timed against SGLang's version of the layer.

## What `submission.py` does

It keeps SGLang's two heavy kernels: FA3 attention and the Marlin W4A16 MoE. It rebuilds everything around them:

| Step | SGLang (baseline) | This submission |
|---|---|---|
| qkv_gate projection | cuBLAS | split-K Triton GEMM with FP32 partials, or cuBLAS (picked in `prepare`) |
| QK-norm, RoPE, FP8 q, FP8 k/v cache store | 5–6 kernels | 1 kernel (`_qkv_post`) that reads the partials |
| per-head output norm × sigmoid(gate) | JIT kernel | 1 kernel (`_norm_gate`) that reads the gate from the qkv partials |
| o projection | cuBLAS | split-K Triton or cuBLAS |
| post-attention norm, residual gate, pre-FFN norm | JIT kernel | 1 kernel (`_attn_boundary`) |
| router (side stream), pooler | 2 GEMMs on 2 streams | 1 GEMM: the router's BF16 hi and lo rows are appended under the pooler |
| routing, local-expert map, pre-expert norm | 4–5 kernels | 1 kernel (`_pool_route`) |
| experts | Marlin | Marlin (unchanged) |
| unpooler, zero pad (the all-gather), post-FFN boundary | GEMM + pad + JIT | GEMM + 1 kernel (`_ffn_boundary`), with the pad folded in |

`prepare()` times each GEMM shape at T = 1, 8, 16 and 32 tokens against cuBLAS, with a cold L2. It charges each split-K candidate for its partials' round trip and keeps the fastest. Every candidate is checked against cuBLAS's result first. Above 32 tokens the GEMMs use cuBLAS.

Precision follows the rules:
- BF16 weights stay BF16, and NVFP4 weights stay NVFP4.
- The router runs as the allowed BF16 hi+lo pair with FP32 accumulation.
- Queries are cast to FP8 E4M3, as FA3 does.
- The cache stays FP8.

## Testing without a GPU

`tests/` runs the Triton kernels in the CPU interpreter (`TRITON_INTERPRET=1`):

- `test_kernels.py` checks each epilogue kernel against PyTorch.
- `test_gemm.py` checks the split-K GEMM with fp32 inputs. The interpreter's BF16 `tl.dot` is broken.
- `test_e2e.py` runs `kernel_fn` end to end through the judge's own `check_implementation`. FA3 and Marlin are replaced by PyTorch versions, and five test-like cases all pass.
- `fp8fix.py` patches an interpreter bug in float→e4m3 rounding. Values that round up into the next exponent come out halved. The GPU's conversion is correct.

Setup:

```bash
okc init muse_decode_layer /tmp/okc        # reference.py, nvfp4.py, checks.py
cp submission.py /tmp/okc/muse_decode_layer/
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install triton numpy
cd tests && OKC_DIR=/tmp/okc/muse_decode_layer python test_e2e.py
```

## Submitting

The rules have to be accepted by a person (`okc login`). Then:

```bash
okc check muse_decode_layer submission.py    # off the board, 20 a day
okc submit muse_decode_layer submission.py   # 10 a day
```

Not yet run on the judge, so there are no timings yet. `prepare()` prints its tuning table, which comes back in the run's log.
