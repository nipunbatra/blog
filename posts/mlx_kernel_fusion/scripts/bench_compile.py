"""Eager vs mx.compile on an element-wise composite (GELU)."""

import math
import time

import mlx.core as mx


def gelu(x: mx.array) -> mx.array:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + mx.tanh(c * (x + 0.044715 * x * x * x)))


gelu_compiled = mx.compile(gelu)


def bench(fn, x, warmup=5, iters=50):
    for _ in range(warmup):
        mx.eval(fn(x))
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(x)
        mx.eval(y)
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    print(f"{'shape':>20} {'eager (ms)':>12} {'compiled (ms)':>14} {'speedup':>9}")
    for shape in [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]:
        x = mx.random.normal(shape)
        mx.eval(x)
        t_e = bench(gelu, x)
        t_c = bench(gelu_compiled, x)
        print(f"{str(shape):>20} {t_e:>12.3f} {t_c:>14.3f} {t_e/t_c:>8.2f}x")
