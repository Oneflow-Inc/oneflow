"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import contextlib
import os
import subprocess
import sys
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest

# Bits in the significand of each type, so that 2 ** -bits bounds one rounding.
_SIGNIFICAND_BITS = {flow.float16: 11, flow.bfloat16: 8}

_K = 1 << 20


def _check(test_case, dtype, a, b, fn, desc):
    rng = np.random.RandomState(0)
    _check_values(
        test_case, dtype, rng.uniform(-1, 1, a), rng.uniform(-1, 1, b), fn, desc
    )


def _check_values(test_case, dtype, a, b, fn, desc):
    # Rounded on the host so that only the 16-bit operands reach the GPU.
    fa = flow.tensor(a).to(dtype)
    fb = flow.tensor(b).to(dtype)
    out = fn(fa.to("cuda"), fb.to("cuda"))
    # The reference multiplies exactly the values the kernel sees.
    ref = fn(fa.to(flow.float64), fb.to(flow.float64)).numpy()
    _check_result(test_case, dtype, out, ref, desc)


def _check_result(test_case, dtype, out, ref, desc):
    test_case.assertEqual(out.dtype, dtype)
    out = _to_numpy(out)
    # inf and NaN in the inputs give the same non-finite results as in the reference.
    for where in [np.isnan, np.isposinf, np.isneginf]:
        test_case.assertTrue(
            np.array_equal(where(out), where(ref)), "%s %s" % (desc, dtype)
        )
    finite = np.isfinite(ref)
    err = np.abs(out - ref)[finite].max() / np.abs(ref[finite]).max()
    # Accumulated at float precision and rounded to dtype once; rounding the partial sums to
    # dtype along k gives several times this.
    tol = 4 * 2.0 ** -_SIGNIFICAND_BITS[dtype]
    test_case.assertLess(err, tol, "%s %s" % (desc, dtype))


def _check_partial_sum_overflow(test_case):
    # A weight gradient x^T @ dy over 8 blocks of 8192 tokens whose sums (about +-98304) exceed
    # float16 but cancel to a result inside it (about 32768), as a single GEMM accumulating in
    # float gives it. Every other input column is small, so its sums stay finite.
    k, block = 1 << 16, 1 << 13
    rng = np.random.RandomState(0)
    x = 4 + rng.uniform(-0.1, 0.1, (k, 64))
    x[:, 1::2] /= 16
    sign = np.where(np.arange(k) // block % 2 == 0, 3.0, -2.75)
    dy = sign[:, None] + rng.uniform(-0.1, 0.1, (k, 32))
    fn = lambda x, y: flow.matmul(x, y, transpose_a=True)
    _check_values(test_case, flow.float16, x, dy, fn, "partial sums beyond float16")
    # Scaled by alpha, the same sums fit in float16 but are several times the result.
    scaled = lambda x, y: flow.matmul(x, y, transpose_a=True, alpha=-0.5)
    _check_values(test_case, flow.float16, x, dy, scaled, "large partial sums")
    x[100, 2] = np.inf
    x[block + 100, 3] = np.inf
    x[300, 4] = np.nan
    _check_values(test_case, flow.float16, x, dy, fn, "inf and NaN inputs")


def _check_rounding_aligned(test_case):
    # Dot products of ones whose 2^15-long chunk sums, all but one of magnitude 2^(bits - 1),
    # lie just short of a rounding midpoint of dtype and so all round the same way: rounded per
    # chunk they lose about 9 units of the result's last place, though no sum is large. The
    # outputs are one element and an 8 x 33 matrix, which is a full block and a narrow one.
    k, chunk = _K, 1 << 15
    for dtype in [flow.float16, flow.bfloat16]:
        bits = _SIGNIFICAND_BITS[dtype]
        base = 2.0 ** (bits - 1)
        step = base / chunk
        dy = np.zeros(k)
        for s in range(k // chunk - 1):
            seg = slice(s * chunk, (s + 1) * chunk)
            if s % 2 == 0 or s >= 28:
                # A chunk sum of base + 1/2 - 2^-bits, which rounds down.
                dy[seg] = step
                dy[s * chunk] = 0.5 - 2.0 ** -bits + step
            else:
                # A chunk sum of -(base + 1/2 + 2^-bits), which rounds down too.
                dy[seg] = -step
                dy[s * chunk] = -0.5 - 2.0 ** -bits - step
        dy[k - chunk : k - chunk + 16] = base / 32
        for m, n in [(1, 1), (8, 33)]:
            _check_values(
                test_case,
                dtype,
                np.ones((m, k)),
                np.repeat(dy[:, None], n, axis=1),
                flow.matmul,
                "chunk sums rounding alike (%d, %d)" % (m, n),
            )


def _check_grouped(test_case, problems, bias):
    # grouped_matmul_bias computes x * w^T + b per problem (m, n, k), in float16 only.
    rng = np.random.RandomState(0)
    half = lambda a: flow.tensor(a).to(flow.float16).to("cuda")
    xs = [half(rng.uniform(-1, 1, (m, k))) for m, n, k in problems]
    ws = [half(rng.uniform(-1, 1, (n, k))) for m, n, k in problems]
    bs = [half(rng.uniform(-1, 1, n)) for m, n, k in problems] if bias else None
    if bias:
        ys = flow._C.grouped_matmul_bias(xs, ws, bs)
    else:
        ys = flow._C.grouped_matmul(xs, ws)
    for i, (x, w, y) in enumerate(zip(xs, ws, ys)):
        test_case.assertEqual(y.dtype, flow.float16)
        test_case.assertEqual(tuple(y.shape), problems[i][:2])
        if y.numel() == 0:
            continue
        ref = x.cpu().to(flow.float64).numpy() @ w.cpu().to(flow.float64).numpy().T
        if bias:
            ref += bs[i].cpu().to(flow.float64).numpy()
        err = np.abs(y.cpu().to(flow.float64).numpy() - ref).max() / np.abs(ref).max()
        tol = 4 * 2.0 ** -_SIGNIFICAND_BITS[flow.float16]
        test_case.assertLess(err, tol, "grouped %s bias=%s" % (problems[i], bias))


def _check_all(test_case):
    for dtype in [flow.float16, flow.bfloat16]:
        _check(test_case, dtype, (2, _K), (_K, 16), flow.matmul, "matmul")
        _check(
            test_case,
            dtype,
            (2, _K),
            (16, _K),
            lambda x, y: flow.matmul(x, y, transpose_b=True),
            "matmul transpose_b",
        )
        # Neither operand has k as its leading dimension.
        _check(
            test_case,
            dtype,
            (_K, 2),
            (_K, 16),
            lambda x, y: flow.matmul(x, y, transpose_a=True),
            "matmul transpose_a",
        )
        _check(test_case, dtype, (2, 1, _K), (2, _K, 16), flow.bmm, "bmm")
        _check(test_case, dtype, (1, _K), (2, _K, 16), flow.matmul, "broadcast")
        _check(test_case, dtype, (16, _K), (_K, 1), flow.matmul, "matrix-vector")
    problems = [(1, 16, _K), (3, 8, _K)]
    _check_grouped(test_case, problems, True)
    _check_grouped(test_case, problems, False)
    # Shorter k, including one that is not a multiple of 2^15.
    for k in [163840, 200000, 1 << 18]:
        for dtype in [flow.float16, flow.bfloat16]:
            _check(
                test_case,
                dtype,
                (2, k),
                (16, k),
                lambda x, y: flow.matmul(x, y, transpose_b=True),
                "matmul transpose_b k=%d" % k,
            )
        _check_grouped(test_case, [(3, 8, k)], False)
    # Empty outputs: a weight or an input without rows.
    for problems in [[(2, 0, _K)], [(0, 2, _K)]]:
        _check_grouped(test_case, problems, False)
    _check_partial_sum_overflow(test_case)
    _check_rounding_aligned(test_case)


def _check_row_tiles(test_case):
    # With tiles of 256 16-bit elements (128 float outputs), outputs with 300 cuBLAS rows (columns
    # of the oneflow output) take three tiles of 100 rows per column. The last two cases run one
    # GEMM per batch.
    gen = flow.Generator(device="cuda")
    gen.manual_seed(0)
    rand = lambda *s: flow.rand(*s, device="cuda", generator=gen) * 2 - 1
    k, n = 1 << 15, 300
    cases = [
        ((3, k), (k, n), {}, "matmul"),
        ((3, k), (n, k), {"transpose_b": True}, "matmul transpose_b"),
        (
            (k, 3),
            (n, k),
            {"transpose_a": True, "transpose_b": True},
            "matmul transpose_a transpose_b",
        ),
        ((3, k), (2, k, n), {}, "broadcast"),
        ((2, 3, k), (2, k, n), {}, "batched"),
    ]
    for dtype in [flow.float16, flow.bfloat16]:
        for a, b, kwargs, desc in cases:
            fa, fb = rand(*a).to(dtype), rand(*b).to(dtype)
            out = flow.matmul(fa, fb, **kwargs)
            test_case.assertEqual(out.dtype, dtype)
            # The reference multiplies exactly the values the kernel sees.
            ref = flow.matmul(fa.to(flow.float32), fb.to(flow.float32), **kwargs)
            err = ((out.to(flow.float32) - ref).abs().max() / ref.abs().max()).item()
            tol = 4 * 2.0 ** -_SIGNIFICAND_BITS[dtype]
            test_case.assertLess(err, tol, "%s, rows in pieces %s" % (desc, dtype))


def _check_narrow_last_tile(test_case):
    # A weight gradient x^T @ dy with an odd number of inputs, cut into output tiles 2 wide, so the
    # last tile is 1 wide: each tile must be written exactly once, whichever library runs it.
    with _env("ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE", "100"):
        _check(
            test_case,
            flow.float16,
            (40000, 1025),
            (40000, 8),
            lambda x, y: flow.matmul(x, y, transpose_a=True),
            "matmul transpose_a, 1-wide last tile",
        )


def _to_numpy(t):
    # Exact for 16-bit values, and much faster than converting them on the host.
    return t.to(flow.float32).cpu().numpy().astype(np.float64)


def _rounded(rng, dtype, *shape, low=-1, high=1):
    # The values of dtype nearest to uniform samples, on the GPU, and as float64 on the host.
    t = flow.tensor(rng.uniform(low, high, shape).astype(np.float32), device="cuda")
    t = t.to(dtype)
    return t, _to_numpy(t)


def _check_same(test_case, outs, desc):
    # On ROCm a fused kernel whose GEMM has a long k runs matmul's GEMM and applies its epilogue
    # afterwards, so it matches the unfused ops exactly; cuBLASLt applies the epilogue before
    # rounding.
    if flow._oneflow_internal.flags.with_rocm():
        test_case.assertTrue(flow.equal(*outs), desc)


def _check_fused_linear(test_case):
    # Linear layers whose inputs are longer than 163840 run fused kernels with k that long.
    rng = np.random.RandomState(0)
    for k in [200000, _K]:
        for dtype in [flow.float16, flow.bfloat16]:
            x, x64 = _rounded(rng, dtype, 2, k)
            w, w64 = _rounded(rng, dtype, 16, k)
            b, b64 = _rounded(rng, dtype, 16)
            c, c64 = _rounded(rng, dtype, 2, 16, low=-100, high=100)
            ref = x64 @ w64.T + b64
            unfused = flow.matmul(x, w, transpose_b=True) + b
            desc = "fused_matmul_bias k=%d" % k
            out = flow._C.fused_matmul_bias(x, w, b)
            _check_result(test_case, dtype, out, ref, desc)
            # The same GEMM as matmul, then the bias.
            _check_same(test_case, (out, unfused), desc)
            # alpha scales the product and beta the addend, each once.
            out = flow._C.fused_matmul_bias(
                x, w, b, _add_to_output=c, alpha=0.5, beta=2
            )
            ref_c = 0.5 * (x64 @ w64.T) + b64 + 2 * c64
            _check_result(test_case, dtype, out, ref_c, desc + " alpha beta")
            with _env("ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR", "1"):
                linear = flow.nn.Linear(k, 16).to("cuda", dtype)
            test_case.assertTrue(linear.use_fused_matmul_bias)
            linear.weight.data = w
            linear.bias.data = b
            out = linear(x)
            _check_result(test_case, dtype, out, ref, "fused Linear k=%d" % k)
            _check_same(test_case, (out, unfused), "fused Linear k=%d" % k)


def _check_fused_mlp_layer(test_case):
    # One layer of the fused MLP kernels, with and without the activation, against matmul and
    # the bias and ReLU applied after it.
    rng = np.random.RandomState(1)
    for dtype in [flow.float16, flow.bfloat16]:
        x, x64 = _rounded(rng, dtype, 2, _K)
        w, w64 = _rounded(rng, dtype, 16, _K)
        b, b64 = _rounded(rng, dtype, 16)
        ref = x64 @ w64.T + b64
        unfused = flow.matmul(x, w, transpose_b=True) + b
        for skip in [True, False]:
            outs = {
                "fused_mlp": flow._C.fused_mlp(x, [w], [b], skip),
                "with dropout": flow._C.fused_matmul_bias_add_relu_dropout(
                    x, [w], [b], skip, [0.0]
                ),
            }
            for name, out in outs.items():
                desc = "%s, final activation skipped: %s" % (name, skip)
                _check_result(
                    test_case, dtype, out, ref if skip else np.maximum(ref, 0), desc
                )
                expected = unfused if skip else flow.relu(unfused)
                _check_same(test_case, (out, expected), desc)


def _mlp_case(rng, dtype, wide_hidden):
    # Two-layer MLPs whose long-k GEMMs are the ones the backward kernels run: k is the input
    # length when the hidden layer is narrow, and the hidden width when it is wide. A wide hidden
    # layer gets integer inputs and weights and half-integer biases, so its ReLU mask is exact.
    k = _K
    if wide_hidden:
        x, x64 = _int_rounded(rng, dtype, (2, 8), 2)
        w0, w0_64 = _int_rounded(rng, dtype, (k, 8), 1)
        b0, b0_64 = _int_rounded(rng, dtype, (k,), 1)
        b0, b0_64 = b0 + 0.5 * flow.sign(b0 - 0.25), b0_64 + 0.5 * np.sign(b0_64 - 0.25)
        w1, w1_64 = _rounded(rng, dtype, 8, k)
        b1, b1_64 = _rounded(rng, dtype, 8)
    else:
        w0, w0_64 = _rounded(rng, dtype, 8, k)
        b0, b0_64 = _rounded(rng, dtype, 8)
        while True:
            # Few enough ReLU inputs, from a long-k GEMM, to draw them all far from zero.
            x, x64 = _rounded(rng, dtype, 2, k)
            pre = np.abs(x64 @ w0_64.T + b0_64)
            if pre.min() > pre.max() / 64:
                break
        w1, w1_64 = _rounded(rng, dtype, k, 8)
        b1, b1_64 = _rounded(rng, dtype, k)
    dy, dy64 = _rounded(rng, dtype, 2, w1.shape[0])
    return [x, w0, b0, w1, b1, dy], [x64, w0_64, b0_64, w1_64, b1_64, dy64]


def _int_rounded(rng, dtype, shape, high):
    t = flow.tensor(
        rng.randint(-high, high + 1, shape).astype(np.float32), device="cuda"
    )
    t = t.to(dtype)
    return t, _to_numpy(t)


def _check_fused_mlp_grad(test_case):
    # The forward and backward of two-layer MLPs through the fused kernels, against float64, in
    # both backward implementations. Their backward kernels have no bfloat16 version.
    rng = np.random.RandomState(2)
    dtype = flow.float16
    for wide_hidden in [False, True]:
        gpu, (x64, w0_64, b0_64, w1_64, b1_64, dy64) = _mlp_case(
            rng, dtype, wide_hidden
        )
        pre = x64 @ w0_64.T + b0_64
        h = np.maximum(pre, 0)
        dpre = (dy64 @ w1_64) * (pre > 0)
        refs = {
            "out": h @ w1_64.T + b1_64,
            "x": dpre @ w0_64,
            "w0": dpre.T @ x64,
            "b0": dpre.sum(0),
            "w1": dy64.T @ h,
            "b1": dy64.sum(0),
        }
        grads = {}
        for name, fused_grad in [
            ("fused_mlp", "0"),
            ("fused_mlp, fused grad", "1"),
            ("with dropout", "0"),
        ]:
            x, w0, b0, w1, b1 = [t.clone().requires_grad_() for t in gpu[:5]]
            dy = gpu[5]
            if name == "with dropout":
                out = flow._C.fused_matmul_bias_add_relu_dropout(
                    x, [w0, w1], [b0, b1], True, [0.0, 0.0]
                )
            else:
                out = flow._C.fused_mlp(x, [w0, w1], [b0, b1], True)
            with _env("ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD", fused_grad):
                out.backward(dy)
            got = {
                "out": out,
                "x": x.grad,
                "w0": w0.grad,
                "b0": b0.grad,
                "w1": w1.grad,
                "b1": b1.grad,
            }
            for key, ref in refs.items():
                desc = "%s %s grad, wide hidden layer: %s" % (name, key, wide_hidden)
                _check_result(test_case, dtype, got[key], ref, desc)
            grads[name] = got
        # The long-k GEMMs of the backward kernels are matmul's: the input gradient, which
        # the other backward computes with matmul, and the hidden layer's bias gradient.
        x, w0, b0, w1, b1, dy = gpu
        mask = (flow.matmul(x, w0, transpose_b=True) + b0) > 0
        unfused = {
            "x": grads["fused_mlp"]["x"],
            "b0": (flow.matmul(dy, w1) * mask).sum(0),
        }
        key = "x" if wide_hidden else "b0"
        for name, got in grads.items():
            desc = "%s %s grad, wide hidden layer: %s" % (name, key, wide_hidden)
            _check_same(test_case, (got[key], unfused[key]), desc)


@contextlib.contextmanager
def _env(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[name]
        else:
            os.environ[name] = old


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestMatmulLongK(flow.unittest.TestCase):
    def test_matmul_long_k(test_case):
        _check_all(test_case)
        _check_narrow_last_tile(test_case)

    def test_fused_matmul_long_k(test_case):
        _check_fused_linear(test_case)
        _check_fused_mlp_layer(test_case)
        _check_fused_mlp_grad(test_case)

    def test_matmul_long_k_without_hipblaslt(test_case):
        # On ROCm, the path taken when hipBLASLt has no algorithm for a shape. Elsewhere the
        # variable has no effect.
        with _env("ONEFLOW_EP_CUDA_LONG_K_GEMM_USE_HIPBLASLT", "0"):
            _check_all(test_case)
            # An output larger than the default 4 MiB workspace holds as float, so the float
            # tile is a separate allocation.
            _check(
                test_case,
                flow.float16,
                (1040, 1 << 15),
                (1024, 1 << 15),
                lambda x, y: flow.matmul(x, y, transpose_b=True),
                "matmul transpose_b, tile larger than the workspace",
            )
            with _env("ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE", "256"):
                _check_row_tiles(test_case)
            _check_narrow_last_tile(test_case)

    def test_matmul_long_k_without_workspace(test_case):
        # With no cuBLAS workspace the float tiles of the fallback need their own memory. A stream
        # reads the workspace size when it is created, so the test runs in a new process.
        env = dict(os.environ, ONEFLOW_EP_CUDA_CUBLAS_WORKSPACE_SIZE_MB="0")
        test = "TestMatmulLongK.test_matmul_long_k_without_hipblaslt"
        result = subprocess.run([sys.executable, __file__, test], env=env)
        test_case.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
