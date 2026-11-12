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
import os
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest


def _conv_bias_grad(dy, data_format):
    # conv_bias_grad has no Python binding; build the op with its attributes and run it
    # through a generic one-input dispatcher (its "name" attribute is not one of the op's).
    builder = flow._oneflow_internal.one.OpBuilder(
        "conv_bias_grad", flow.framework.id_util.UniqueStr("conv_bias_grad")
    )
    builder.input("dy", 1)
    builder.output("bias_diff", 1)
    builder.attr("data_format", 'at_string: "%s"' % data_format)
    builder.attr("num_spatial_dims", "at_int32: %d" % (dy.ndim - 2))
    return flow._C.dispatch_ofrecord_bytes_decoder(builder.build(), dy, name="")


# Bits in the significand of each type, so that 2 ** -bits bounds one rounding.
_SIGNIFICAND_BITS = {
    flow.float16: 11,
    flow.bfloat16: 8,
    flow.float32: 24,
    flow.float64: 53,
}


def _test_conv_bias_grad(test_case, dtype, data_format, shape):
    rng = np.random.RandomState(0)
    dy = flow.tensor(rng.randn(*shape), dtype=dtype, device="cuda")
    # The reference sums exactly the values the kernel sees.
    x = dy.to(flow.float64).numpy()
    channel_axis = 1 if data_format == "channels_first" else len(shape) - 1
    axes = tuple(i for i in range(len(shape)) if i != channel_axis)
    ref = x.sum(axis=axes)
    out = _conv_bias_grad(dy, data_format)
    test_case.assertEqual(out.dtype, dtype)
    test_case.assertEqual(tuple(out.shape), (shape[channel_axis],))
    got = out.to(flow.float64).numpy()
    # Summed at float precision and rounded to dtype once.
    eps = 2.0 ** -_SIGNIFICAND_BITS[dtype]
    accumulate_eps = 2.0 ** -_SIGNIFICAND_BITS[flow.float32]
    if dtype == flow.float64:
        accumulate_eps = eps
    tol = eps * np.abs(ref) + 16 * accumulate_eps * np.abs(x).sum(axis=axes)
    err = np.abs(got - ref)
    test_case.assertTrue(
        np.all(err <= tol),
        "%s %s %s: max err %g, max tol %g"
        % (dtype, data_format, shape, err.max(), tol[np.argmax(err - tol)]),
    )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestConvBiasGrad(flow.unittest.TestCase):
    def test_conv_bias_grad(test_case):
        for dtype in [flow.float16, flow.bfloat16, flow.float32, flow.float64]:
            for data_format in ["channels_first", "channels_last"]:
                for shape in [(256, 64, 7, 7), (5, 3, 9), (2, 4, 3, 5, 6)]:
                    if data_format == "channels_last":
                        shape = (shape[0],) + shape[2:] + (shape[1],)
                    _test_conv_bias_grad(test_case, dtype, data_format, shape)

    def test_conv_bias_grad_empty_batch(test_case):
        for data_format in ["channels_first", "channels_last"]:
            shape = (0, 4, 3, 3) if data_format == "channels_first" else (0, 3, 3, 4)
            dy = flow.zeros(shape, dtype=flow.float16, device="cuda")
            out = _conv_bias_grad(dy, data_format)
            test_case.assertTrue(np.array_equal(out.numpy(), np.zeros(4, np.float16)))


if __name__ == "__main__":
    unittest.main()
