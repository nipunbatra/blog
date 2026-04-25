"""A handwritten Metal kernel for SiLU + a tiny correctness/perf check."""

import time

import mlx.core as mx


# Element-wise SiLU as a Metal kernel: y = x * sigmoid(x)
silu_kernel = mx.fast.metal_kernel(
    name="silu_kernel",
    input_names=["x"],
    output_names=["y"],
    source="""
        uint tid = thread_position_in_grid.x;
        if (tid >= x_shape[0]) return;
        float xv = x[tid];
        y[tid] = xv / (1.0f + metal::exp(-xv));
    """,
)


def silu_metal(x):
    flat = x.reshape(-1)
    (out,) = silu_kernel(
        inputs=[flat],
        grid=(flat.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[flat.dtype],
    )
    return out.reshape(x.shape)


def silu_python(x):
    return x * mx.sigmoid(x)


silu_compiled = mx.compile(silu_python)


def bench(fn, x, warmup=5, iters=50):
    for _ in range(warmup):
        mx.eval(fn(x))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(x))
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    # Correctness
    x = mx.random.normal((1024,))
    a = silu_python(x)
    b = silu_metal(x)
    print("max |py - metal| =", float(mx.abs(a - b).max()))

    print(f"\n{'shape':>20} {'python':>10} {'compiled':>10} {'metal':>10}")
    for shape in [(1024 * 1024,), (16 * 1024 * 1024,), (64 * 1024 * 1024,)]:
        x = mx.random.normal(shape)
        mx.eval(x)
        t_p = bench(silu_python, x)
        t_c = bench(silu_compiled, x)
        t_m = bench(silu_metal, x)
        print(f"{str(shape):>20} {t_p:>10.3f} {t_c:>10.3f} {t_m:>10.3f}")
