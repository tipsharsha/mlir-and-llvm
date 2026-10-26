"""Test-only: make the Triton CPU interpreter's float32 -> e4m3 cast round like the GPU's."""
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
