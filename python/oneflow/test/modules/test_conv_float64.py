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
import itertools
import os
import unittest

import numpy as np
import torch

import oneflow as flow
import oneflow.unittest

_CONV = {1: flow._C.conv1d, 2: flow._C.conv2d, 3: flow._C.conv3d}
_DECONV = {1: flow._C.deconv1d, 2: flow._C.deconv2d, 3: flow._C.deconv3d}
_TORCH_CONV = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}
_TORCH_DECONV = {
    1: torch.nn.functional.conv_transpose1d,
    2: torch.nn.functional.conv_transpose2d,
    3: torch.nn.functional.conv_transpose3d,
}
_SPATIAL = {1: (9,), 2: (7, 8), 3: (5, 6, 4)}
_KERNEL = {1: (3,), 2: (3, 2), 3: (2, 3, 2)}
# (stride, padding, dilation)
_GEOMETRY = [(1, 0, 1), (2, 1, 1), (1, 2, 2), (3, 1, 2)]


def _attr(value):
    if isinstance(value, str):
        return 'at_string: "%s"' % value
    if isinstance(value, list):
        return "at_list_int32 { %s }" % " ".join("val: %d" % v for v in value)
    return "at_int32: %d" % value


def _apply_op(op_type, inputs, output, attrs, tensors):
    builder = flow._oneflow_internal.one.OpBuilder(
        op_type, flow.framework.id_util.UniqueStr(op_type)
    )
    for name in inputs:
        builder.input(name, 1)
    builder.output(output, 1)
    for name, value in attrs.items():
        builder.attr(name, _attr(value))
    # Any dispatcher taking a TensorTuple applies the op; its extra attrs are not attrs of
    # these ops and are ignored.
    return flow._C.dispatch_crop_mirror_normalize_from_uint8(
        builder.build(), tensors, mean=[0.0], std=[1.0]
    )


def _per_axis(value, ndims):
    return list(value) if isinstance(value, (list, tuple)) else [value] * ndims


def _run(fn, x, w, b, dy, weight_grad):
    # x, w, b and dy are channels_first numpy arrays; fn takes and returns channels_first
    # tensors of oneflow on the GPU, or of torch on the CPU, the reference.
    if fn.reference:
        t = lambda a, grad=True: torch.tensor(a, requires_grad=grad)
    else:
        t = lambda a, grad=True: flow.tensor(a, device="cuda", requires_grad=grad)
    x, w = t(x), t(w, weight_grad)
    b = None if b is None else t(b)
    y = fn(x, w, b)
    (
        y * (torch.tensor(dy) if fn.reference else flow.tensor(dy, device="cuda"))
    ).sum().backward()
    grads = (
        [x.grad] + ([w.grad] if weight_grad else []) + ([] if b is None else [b.grad])
    )
    return [a.detach().numpy() if fn.reference else a.numpy() for a in [y] + grads]


def _compare(test_case, fn, ref_fn, x, w, b, y_shape, desc, weight_grad=True):
    fn.reference = False
    ref_fn.reference = True
    dy = np.random.RandomState(1).randn(*y_shape)
    got = _run(fn, x, w, b, dy, weight_grad)
    ref = _run(ref_fn, x, w, b, dy, weight_grad)
    names = ["y", "dx"] + (["dw"] if weight_grad else []) + ["db"]
    for name, g, r in zip(names, got, ref):
        test_case.assertEqual(g.shape, r.shape, "%s %s" % (desc, name))
        err = np.abs(g - r).max()
        scale = max(np.abs(r).max(), 1.0)
        test_case.assertLess(
            err, 1e-12 * scale, "%s %s: max err %g" % (desc, name, err)
        )


def _test_conv(
    test_case, ndims, channels_last, groups, geometry, bias, weight_grad=True
):
    stride, padding, dilation = (_per_axis(v, ndims) for v in geometry)
    rng = np.random.RandomState(ndims * 100 + groups)
    c, f = (6 if groups == 3 else 4), 6
    x = rng.randn(*((2, c) + _SPATIAL[ndims]))
    w = rng.randn(*((f, c // groups) + _KERNEL[ndims]))
    b = rng.randn(f) if bias else None
    data_format = "channels_last" if channels_last else "channels_first"

    def ref_fn(x, w, b):
        return _TORCH_CONV[ndims](x, w, b, stride, padding, dilation, groups)

    def fn(x, w, b):
        if channels_last:
            # Run on the channels_last layouts, compare in channels_first.
            perm = [0] + list(range(2, ndims + 2)) + [1]
            x, w = x.permute(*perm), w.permute(*perm)
        if b is None:
            y = _CONV[ndims](
                x,
                w,
                None,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channel_pos=data_format,
            )
        else:
            # Give the bias to the convolution op itself, as the fused conv bias path does.
            attrs = {
                "filters": f,
                "kernel_size": list(_KERNEL[ndims]),
                "padding_before": padding,
                "strides": stride,
                "dilation_rate": dilation,
                "groups": groups,
                "data_format": data_format,
            }
            y = _apply_op(
                "conv%dd" % ndims, ["in", "weight", "bias"], "out", attrs, [x, w, b]
            )
        return y.permute(0, ndims + 1, *range(1, ndims + 1)) if channels_last else y

    y_shape = tuple(ref_fn(torch.tensor(x), torch.tensor(w), None).shape)
    desc = "conv%dd %s groups=%d %s bias=%s" % (
        ndims,
        data_format,
        groups,
        geometry,
        bias,
    )
    _compare(test_case, fn, ref_fn, x, w, b, y_shape, desc, weight_grad)


def _test_deconv(test_case, ndims, groups, geometry, bias):
    stride, padding, dilation = geometry
    rng = np.random.RandomState(ndims * 100 + groups + 7)
    c, f = 4, 6
    output_padding = 1 if stride > 1 else 0
    x = rng.randn(*((2, c) + _SPATIAL[ndims]))
    w = rng.randn(*((c, f // groups) + _KERNEL[ndims]))
    b = rng.randn(f) if bias else None

    def ref_fn(x, w, b):
        return _TORCH_DECONV[ndims](
            x, w, b, stride, padding, output_padding, groups, dilation
        )

    def fn(x, w, b):
        return _DECONV[ndims](
            x,
            w,
            b,
            stride=[stride] * ndims,
            padding=[padding] * ndims,
            output_padding=[output_padding] * ndims,
            groups=groups,
            dilation=[dilation] * ndims,
        )

    y_shape = tuple(ref_fn(torch.tensor(x), torch.tensor(w), None).shape)
    desc = "deconv%dd groups=%d %s bias=%s" % (ndims, groups, geometry, bias)
    _compare(test_case, fn, ref_fn, x, w, b, y_shape, desc)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestConvFloat64(flow.unittest.TestCase):
    def test_conv_float64(test_case):
        for i, (ndims, channels_last, groups, geometry) in enumerate(
            itertools.product([1, 2, 3], [False, True], [1, 2], _GEOMETRY)
        ):
            if channels_last and groups > 1:
                # conv_filter_grad supports groups only for channels_first.
                continue
            _test_conv(test_case, ndims, channels_last, groups, geometry, i % 2 == 0)

    def test_conv_float64_channels_last_groups(test_case):
        # Forward and data grad only: conv_filter_grad supports groups only for channels_first.
        for i, (ndims, groups, geometry) in enumerate(
            itertools.product([1, 2, 3], [2, 3], _GEOMETRY)
        ):
            _test_conv(test_case, ndims, True, groups, geometry, i % 2 == 0, False)

    def test_conv_float64_per_axis_geometry(test_case):
        # Different stride, padding and dilation along each spatial axis.
        geometries = {
            2: [((1, 2), (2, 0), (1, 2)), ((3, 1), (0, 1), (2, 1))],
            3: [((2, 1, 3), (0, 1, 2), (1, 2, 1)), ((1, 3, 2), (2, 0, 1), (2, 1, 1))],
        }
        for ndims, cases in geometries.items():
            for i, (geometry, channels_last, groups) in enumerate(
                itertools.product(cases, [False, True], [1, 2])
            ):
                weight_grad = not (channels_last and groups > 1)
                _test_conv(
                    test_case,
                    ndims,
                    channels_last,
                    groups,
                    geometry,
                    i % 2 == 0,
                    weight_grad,
                )

    def test_deconv_float64(test_case):
        for i, (ndims, groups, geometry) in enumerate(
            itertools.product([1, 2, 3], [1, 2], _GEOMETRY)
        ):
            _test_deconv(test_case, ndims, groups, geometry, i % 2 == 1)

    def test_conv_float64_module(test_case):
        # A larger convolution through nn.Conv2d, up to depthwise groups.
        for groups in [1, 4, 16]:
            rng = np.random.RandomState(groups)
            x = rng.randn(8, 16, 32, 32)
            conv = (
                flow.nn.Conv2d(16, 32, 3, padding=1, groups=groups).double().to("cuda")
            )
            y = conv(flow.tensor(x, device="cuda")).numpy()
            ref = torch.nn.functional.conv2d(
                torch.tensor(x),
                torch.tensor(conv.weight.numpy()),
                torch.tensor(conv.bias.numpy()),
                padding=1,
                groups=groups,
            ).numpy()
            err = np.abs(y - ref).max()
            test_case.assertLess(err, 1e-12 * np.abs(ref).max(), "groups=%d" % groups)

    def test_conv_float64_add_to_output(test_case):
        # The kernels add into the output when given _add_to_output. No pass fuses that input
        # into a convolution, so build the ops with it directly.
        rng = np.random.RandomState(3)
        x = rng.randn(2, 4, 9, 9)
        w = rng.randn(6, 2, 3, 3)
        residual = rng.randn(2, 6, 5, 5)
        dy = rng.randn(2, 6, 5, 5)
        dx_residual = rng.randn(2, 4, 9, 9)
        attrs = {
            "padding_before": [1, 1],
            "data_format": "channels_first",
            "kernel_size": [3, 3],
            "strides": [2, 2],
            "dilation_rate": [1, 1],
            "groups": 2,
        }
        cuda = lambda a: flow.tensor(a, device="cuda")
        y = _apply_op(
            "conv2d",
            ["in", "weight", "_add_to_output"],
            "out",
            dict(attrs, filters=6),
            [cuda(x), cuda(w), cuda(residual)],
        )
        dx = _apply_op(
            "conv_data_grad",
            ["dy", "filter", "x_like", "_add_to_output"],
            "dx",
            dict(attrs, num_spatial_dims=2),
            [cuda(dy), cuda(w), cuda(x), cuda(dx_residual)],
        )
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.nn.functional.conv2d(tx, torch.tensor(w), None, 2, 1, 1, 2)
        ty.backward(torch.tensor(dy))
        ref_y = ty.detach().numpy() + residual
        ref_dx = tx.grad.numpy() + dx_residual
        test_case.assertLess(
            np.abs(y.numpy() - ref_y).max(), 1e-12 * np.abs(ref_y).max()
        )
        test_case.assertLess(
            np.abs(dx.numpy() - ref_dx).max(), 1e-12 * np.abs(ref_dx).max()
        )


if __name__ == "__main__":
    unittest.main()
