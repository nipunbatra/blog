"""Composite LayerNorm vs mx.fast.layer_norm vs compiled composite."""

import time

import mlx.core as mx


def layernorm_composite(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) * mx.rsqrt(var + eps)
    return x_hat * weight + bias


layernorm_compiled = mx.compile(layernorm_composite)


def layernorm_fast(x, weight, bias, eps=1e-5):
    return mx.fast.layer_norm(x, weight, bias, eps)


def bench(fn, args, warmup=5, iters=50):
    for _ in range(warmup):
        mx.eval(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    print(f"{'shape':>22} {'composite':>11} {'compiled':>10} {'mx.fast':>9} "
          f"{'fast vs eager':>15}")
    for B, T, D in [(1, 1024, 1024), (1, 2048, 2048), (1, 4096, 4096),
                    (1, 8192, 4096), (4, 4096, 4096)]:
        x = mx.random.normal((B, T, D))
        w = mx.ones((D,))
        b = mx.zeros((D,))
        mx.eval(x, w, b)
        t_e = bench(layernorm_composite, (x, w, b))
        t_c = bench(layernorm_compiled, (x, w, b))
        t_f = bench(layernorm_fast, (x, w, b))
        print(f"{str((B,T,D)):>22} {t_e:>10.3f} {t_c:>9.3f} {t_f:>8.3f} "
              f"{t_e/t_f:>14.2f}x")
