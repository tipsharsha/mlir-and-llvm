"""Test-only fixes to the Triton CPU interpreter: float32 -> e4m3 casts that round like the GPU, and BF16 dots."""
import numpy as np, torch
import triton.language as tl
from triton.runtime import interpreter as I
_orig = I._convert_float
def _fixed(input, input_dtype, output_dtype, rounding_mode):
    if output_dtype == tl.float8e4nv and input_dtype in (tl.float32, tl.float16, tl.bfloat16):
        x = torch.from_numpy(np.ascontiguousarray(input).view(np.float32) if input_dtype == tl.float32
                             else _orig(input, input_dtype, tl.float32, None).view(np.float32))
        x = x.clamp(-448, 448)  # cvt.rn.satfinite
        return x.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    return _orig(input, input_dtype, output_dtype, rounding_mode)
I._convert_float = _fixed


# The interpreter keeps BF16 as raw uint16 bits and its dot multiplies those bits as integers.
_orig_dot = I.InterpreterBuilder.create_dot


def _bf16_to_f32(h):
    if h.dtype == tl.bfloat16:
        return I.TensorHandle((h.data.astype(np.uint32) << 16).view(np.float32), tl.float32)
    return h


def _dot(self, a, b, d, input_precision, max_num_imprecise_acc):
    return _orig_dot(self, _bf16_to_f32(a), _bf16_to_f32(b), d, input_precision, max_num_imprecise_acc)


I.InterpreterBuilder.create_dot = _dot
