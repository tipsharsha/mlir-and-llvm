import os, sys; os.environ["TRITON_INTERPRET"]="1"
sys.path.insert(0, os.environ.get("OKC_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, submission as S
torch.manual_seed(0)
for (m, n, k, cfg) in [(1, 64, 512, (16, 32, 128, 2, 3, 4)), (5, 128, 1024, (16, 64, 256, 4, 3, 4)),
                       (20, 64, 512, (32, 32, 128, 1, 3, 4)), (33, 32, 256, (32, 32, 128, 2, 3, 4))]:
    x = torch.randn(m, k); w = torch.randn(n, k)
    got = S._mm(x, w, cfg)
    print(m, n, k, tuple(got.shape), float((got.sum(0) - x @ w.T).norm() / (x @ w.T).norm()))
print(S._candidates(1, 2304, 8192)[:5], len(S._candidates(1, 2304, 8192)), len(S._candidates(32, 4608, 8192)),
      len(S._candidates(8, 1024, 4096)), len(S._candidates(16, 8192, 1024)))
