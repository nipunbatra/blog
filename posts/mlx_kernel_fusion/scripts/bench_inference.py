"""Token-by-token decode loop: eager vs mx.compile.

This is the inference scenario where compile pays off most: the same tiny
graph runs once per generated token, so compilation amortizes and Python
overhead per step gets absorbed into a single dispatch.
"""

import math
import time

import mlx.core as mx
import mlx.nn as nn


class Block(nn.Module):
    def __init__(self, d, n_heads, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

    def __call__(self, x):
        h = self.ln1(x)
        B, T, D = x.shape
        q = self.q(h).reshape(B, T, self.n_heads, self.d_head).swapaxes(1, 2)
        k = self.k(h).reshape(B, T, self.n_heads, self.d_head).swapaxes(1, 2)
        v = self.v(h).reshape(B, T, self.n_heads, self.d_head).swapaxes(1, 2)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        a = a.swapaxes(1, 2).reshape(B, T, D)
        x = x + self.o(a)
        x = x + self.ff2(nn.gelu(self.ff1(self.ln2(x))))
        return x


def step_factory(model):
    def step(x):
        return model(x)
    return step


class Stack(nn.Module):
    """N stacked transformer blocks — better captures inference-time graph
    fusion since `mx.compile` can fuse across layer boundaries."""

    def __init__(self, n_layers, d, n_heads, ff):
        super().__init__()
        self.blocks = [Block(d, n_heads, ff) for _ in range(n_layers)]

    def __call__(self, x):
        for b in self.blocks:
            x = b(x)
        return x


if __name__ == "__main__":
    print(f"{'(layers,B,T,D,H,FF)':>26} {'eager (ms/tok)':>15} "
          f"{'compiled (ms/tok)':>18} {'speedup':>9}")
    for L, B, T, D, H, FF in [
        (4, 1, 1, 512, 8, 2048),
        (8, 1, 1, 512, 8, 2048),
        (8, 1, 1, 768, 12, 3072),
        (12, 1, 1, 768, 12, 3072),
    ]:
        model = Stack(L, D, H, FF)
        x = mx.random.normal((B, T, D))
        mx.eval(x, model.parameters())

        eager = step_factory(model)
        compiled = mx.compile(step_factory(model))

        for _ in range(20):
            mx.eval(eager(x))
            mx.eval(compiled(x))

        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            mx.eval(eager(x))
        t_e = (time.perf_counter() - t0) / N * 1e3

        t0 = time.perf_counter()
        for _ in range(N):
            mx.eval(compiled(x))
        t_c = (time.perf_counter() - t0) / N * 1e3

        cfg = (L, B, T, D, H, FF)
        print(f"{str(cfg):>26} {t_e:>15.3f} {t_c:>18.3f} {t_e/t_c:>8.2f}x")
