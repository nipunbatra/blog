"""Composite attention vs mx.fast.scaled_dot_product_attention (Flash-style)."""

import math
import time

import mlx.core as mx


def attention_composite(q, k, v, scale):
    # q, k, v: (B, H, T, D)
    scores = (q * scale) @ k.swapaxes(-1, -2)        # (B, H, T, T)  -- materialised
    weights = mx.softmax(scores, axis=-1)            # (B, H, T, T)  -- materialised
    return weights @ v                               # (B, H, T, D)


attention_compiled = mx.compile(attention_composite)


def attention_fast(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def bench(fn, args, warmup=3, iters=20):
    for _ in range(warmup):
        mx.eval(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    B, H, D = 1, 16, 64
    print(f"{'seq len T':>10} {'composite':>11} {'compiled':>10} {'fast SDPA':>11} "
          f"{'fast vs comp':>14}  {'attn matrix MB':>15}")
    for T in [256, 512, 1024, 2048, 4096, 8192]:
        q = mx.random.normal((B, H, T, D))
        k = mx.random.normal((B, H, T, D))
        v = mx.random.normal((B, H, T, D))
        scale = 1.0 / math.sqrt(D)
        mx.eval(q, k, v)
        attn_mb = B * H * T * T * 4 / 1e6
        try:
            t_c = bench(attention_composite, (q, k, v, scale))
        except Exception as e:
            t_c = float("nan")
        try:
            t_cc = bench(attention_compiled, (q, k, v, scale))
        except Exception:
            t_cc = float("nan")
        t_f = bench(attention_fast, (q, k, v, scale))
        ratio = t_c / t_f if t_c == t_c else float("nan")
        print(f"{T:>10} {t_c:>11.3f} {t_cc:>10.3f} {t_f:>11.3f} {ratio:>13.2f}x "
              f"{attn_mb:>14.1f}")
