"""The judge's 13 tests and the scored shapes on any CUDA GPU with Triton (no SGLang needed):
the Triton GEMMs, attention and experts are forced on, each case is checked eagerly with the
judge's check_implementation, and then as a CUDA graph replayed on a fresh input, as the judge
times it. Run from this directory with OKC_DIR pointing at `okc init`'s folder:

    OKC_DIR=/path/to/muse_decode_layer python test_gpu.py [--scored] [--skip-big]

--scored also runs the 12 scored shapes (80K-token contexts; needs ~4 GB more). --skip-big skips
the 300K and 1M-token tests (for GPUs with little memory). Timings printed here are for this GPU
only; they say nothing about an H200."""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("OKC_DIR", ".")]  # this submission, then okc init's reference.py and nvfp4.py
import torch
import submission as S
import reference as R

assert torch.cuda.is_available(), "needs a CUDA GPU"
S.flash_attn_with_kvcache = None  # the Triton attention everywhere
S._marlin_bank = lambda weights: None
S._tune = lambda mats, device: {(name, m): S._candidates(m, *w.shape)[0]
                                for name, (w, ms) in mats.items() for m in ms}
S._tune_moe = lambda state, device: {m: (32, 64, 2, 4) for m in S.MOE_TUNE_TOKENS}
S._tune_attn = lambda device: {}

TESTS = """batch 1, ctx 10, layer sliding
batch 1, ctx 3000, layer full
batch 8, ctx 2047, layer sliding
batch 9, ctx 5000, layer sliding, spread mixed, page 1
batch 3, ctx 6000, layer full, q_len 8, dist outliers
batch 5, ctx 5000, layer sliding, q_len 8, spread mixed, page 1
batch 33, ctx 20000, layer full, spread mixed, page 1
batch 2, ctx 300000, layer sliding, q_len 8
batch 1, ctx 1048576, layer full, q_len 8
batch 4, ctx 7000, layer sliding, q_len 8, page 1, kv_scale 0.015625
batch 1, ctx 80000, layer full, dist soft, page 1
batch 4, ctx 6000, layer sliding, q_len 8, dist soft
batch 16, ctx 20000, layer full, spread mixed, dist soft, page 1"""
SCORED = [f"batch {b}, ctx 80000, layer {layer}, page 1{', q_len 8' if q else ''}"
          f"{', spread agentic' if b > 1 else ''}, dist soft"
          for q in (0, 1) for b in (1, 16, 32) for layer in ("sliding", "full")]


def parse(line):
    case = {}
    for part in line.split(", "):
        key, value = part.split(" ")
        case[key] = value if key in ("layer", "spread", "dist") else float(value) if "." in value else int(value)
    return case


def check(data, out, weights):
    ok, msg = R.check_implementation(data, R.collect(data, out), weights)
    return ("PASS " if ok else "FAIL ") + msg


def main():
    big = "--skip-big" not in sys.argv
    lines = TESTS.splitlines() + (SCORED if "--scored" in sys.argv else [])
    weights = R.generate_weights(R.WEIGHTS_SEED, device="cuda")
    state = S.prepare(weights)
    failed = 0
    for i, line in enumerate(lines):
        case = parse(line)
        if not big and case["ctx"] > 100_000:
            continue
        case["seed"] = 1000 + i
        case["local_start"] = R.case_fields(case, weights)["local_start"]
        data = R.generate_input(**case)
        eager = check(data, S.kernel_fn(data, state), weights)
        # As the judge times it: capture once, then replay on a fresh input of the same shapes.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = S.kernel_fn(data, state)
        fresh = R.generate_input(**{**case, "seed": case["seed"] + 7})
        for key, value in fresh.items():
            if isinstance(value, torch.Tensor):
                data[key].copy_(value)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        graph.replay()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        # The second replay wrote the same cache rows again; outputs are of the fresh input.
        replay = check(data, out, weights)
        failed += not (eager.startswith("PASS") and replay.startswith("PASS"))
        print(f"{line}\n    eager:  {eager}\n    replay: {replay}  ({start.elapsed_time(end) * 1e3:.0f} us here)")
        del graph, data, fresh, out
        torch.cuda.empty_cache()
    print("all passed" if not failed else f"{failed} failed")


if __name__ == "__main__":
    main()
