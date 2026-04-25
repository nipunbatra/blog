"""Training step: small transformer block, eager vs compiled."""

import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


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
        B, T, D = x.shape
        h = self.ln1(x)
        q = self.q(h).reshape(B, T, self.n_heads, self.d_head).swapaxes(1, 2)
        k = self.k(h).reshape(B, T, self.n_heads, self.d_head).swapaxes(1, 2)
        v = self.v(h).reshape(B, T, self.n_heads, self.d_head).swapaxes(1, 2)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        a = a.swapaxes(1, 2).reshape(B, T, D)
        x = x + self.o(a)
        x = x + self.ff2(nn.gelu(self.ff1(self.ln2(x))))
        return x


def loss_fn(model, x, y):
    return ((model(x) - y) ** 2).mean()


def make_step(model, opt, compiled):
    grad_fn = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = grad_fn(model, x, y)
        opt.update(model, grads)
        return loss

    if compiled:
        # Capture the model + optimizer state so updates persist across calls
        state = [model.state, opt.state, mx.random.state]
        step = mx.compile(step, inputs=state, outputs=state)
    return step


def bench(step, x, y, warmup=3, iters=20):
    for _ in range(warmup):
        mx.eval(step(x, y))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(step(x, y))
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    print(f"{'shape (B,T,D,H,FF)':>26} {'eager (ms)':>12} {'compiled (ms)':>14} "
          f"{'speedup':>9}")
    for cfg in [(2, 256, 512, 8, 2048), (2, 512, 512, 8, 2048),
                (2, 1024, 768, 12, 3072), (2, 2048, 768, 12, 3072)]:
        B, T, D, H, FF = cfg
        x = mx.random.normal((B, T, D))
        y = mx.random.normal((B, T, D))
        mx.eval(x, y)
        m1 = Block(D, H, FF)
        m2 = Block(D, H, FF)
        # Match initial weights so timings are apples to apples
        m2.update(m1.parameters())
        opt1 = optim.AdamW(1e-4)
        opt2 = optim.AdamW(1e-4)
        opt1.init(m1.parameters())
        opt2.init(m2.parameters())
        step_e = make_step(m1, opt1, compiled=False)
        step_c = make_step(m2, opt2, compiled=True)
        t_e = bench(step_e, x, y)
        t_c = bench(step_c, x, y)
        print(f"{str(cfg):>26} {t_e:>12.3f} {t_c:>14.3f} {t_e/t_c:>8.2f}x")
