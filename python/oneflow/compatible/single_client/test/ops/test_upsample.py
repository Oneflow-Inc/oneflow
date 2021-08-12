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
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def cartesian(arrays, out=None):
    """
    From https://stackoverflow.com/a/1235363
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def interpolate_1d_with_x(
    data,
    scale_factor,
    x,
    get_coeffs,
    roi=None,
    extrapolation_value=0.0,
    scaler="half_pixel",
    exclude_outside=False,
):
    def get_neighbor_idxes(x, n, limit):
        """
        Return the n nearest indexes, prefer the indexes smaller than x
        As a result, the ratio must be in (0, 1]
        Examples:
        get_neighbor_idxes(4, 2, 10) == [3, 4]
        get_neighbor_idxes(4, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.4, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.5, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.6, 3, 10) == [4, 5, 6]
        get_neighbor_idxes(4.4, 1, 10) == [4]
        get_neighbor_idxes(4.6, 1, 10) == [5]
        :param x:
        :param n: the number of the wanted indexes
        :param limit: the maximum value of index
        :return: An np.array containing n nearest indexes in ascending order
        """
        idxes = sorted(range(limit), key=lambda idx: (abs(x - idx), idx))[:n]
        idxes = sorted(idxes)
        return np.array(idxes)

    def get_neighbor(x, n, data):
        """
        Pad `data` in 'edge' mode, and get n nearest elements in the padded array and their indexes in the original
        array
        :param x:
        :param n:  the number of the wanted elements
        :param data: the array
        :return: A tuple containing the indexes of neighbor elements (the index can be smaller than 0 or higher than
        len(data)) and the value of these elements
        """
        pad_width = np.ceil(n / 2).astype(np.int32)
        padded = np.pad(data, pad_width, mode="edge")
        x += pad_width
        idxes = get_neighbor_idxes(x, n, len(padded))
        ret = padded[idxes]
        return (idxes - pad_width, ret)

    input_width = len(data)
    output_width = scale_factor * input_width
    if scaler == "align_corners":
        if output_width == 1:
            x_ori = 0.0
        else:
            x_ori = x * (input_width - 1) / (output_width - 1)
    elif scaler == "asymmetric":
        x_ori = x / scale_factor
    elif scaler == "pytorch_half_pixel":
        if output_width == 1:
            x_ori = -0.5
        else:
            x_ori = (x + 0.5) / scale_factor - 0.5
    else:
        x_ori = (x + 0.5) / scale_factor - 0.5
    x_ori_int = np.floor(x_ori).astype(np.int32).item()
    if x_ori.is_integer():
        ratio = 1
    else:
        ratio = x_ori - x_ori_int
    coeffs = get_coeffs(ratio)
    n = len(coeffs)
    (idxes, points) = get_neighbor(x_ori, n, data)
    if exclude_outside:
        for (i, idx) in enumerate(idxes):
            if idx < 0 or idx >= input_width:
                coeffs[i] = 0
        coeffs /= sum(coeffs)
    return np.dot(coeffs, points).item()


def interpolate_nd_with_x(data, n, scale_factors, x, get_coeffs, roi=None, **kwargs):
    if n == 1:
        return interpolate_1d_with_x(
            data, scale_factors[0], x[0], get_coeffs, roi=roi, **kwargs
        )
    return interpolate_1d_with_x(
        [
            interpolate_nd_with_x(
                data[i],
                n - 1,
                scale_factors[1:],
                x[1:],
                get_coeffs,
                roi=None if roi is None else np.concatenate([roi[1:n], roi[n + 1 :]]),
                **kwargs
            )
            for i in range(data.shape[0])
        ],
        scale_factors[0],
        x[0],
        get_coeffs,
        roi=None if roi is None else [roi[0], roi[n]],
        **kwargs
    )


def interpolate_nd(
    data, get_coeffs, output_size=None, scale_factors=None, roi=None, **kwargs
):
    def get_all_coords(data):
        return cartesian([list(range(data.shape[i])) for i in range(len(data.shape))])

    assert output_size is not None or scale_factors is not None
    if output_size is not None:
        scale_factors = np.array(output_size) / np.array(data.shape)
    else:
        if isinstance(scale_factors, int):
            height_scale = scale_factors
            width_scale = scale_factors
        else:
            assert isinstance(scale_factors, (list, tuple))
            assert len(scale_factors) == 2
            height_scale = scale_factors[0]
            width_scale = scale_factors[1]
        output_size = np.stack(
            [
                data.shape[0],
                data.shape[1],
                data.shape[2] * height_scale,
                data.shape[3] * width_scale,
            ]
        ).astype(np.int32)
        scale_factors = np.array([1, 1, height_scale, width_scale])
    assert scale_factors is not None
    ret = np.zeros(output_size)
    for x in get_all_coords(ret):
        ret[tuple(x)] = interpolate_nd_with_x(
            data, len(data.shape), scale_factors, x, get_coeffs, roi=roi, **kwargs
        )
    return ret


def linear_coeffs(ratio):
    return np.array([1 - ratio, ratio])


def nearest_coeffs(ratio, mode="round_prefer_floor"):
    if type(ratio) == int or ratio.is_integer():
        return np.array([0, 1])
    elif mode == "round_prefer_floor":
        return np.array([ratio <= 0.5, ratio > 0.5])
    elif mode == "round_prefer_ceil":
        return np.array([ratio < 0.5, ratio >= 0.5])
    elif mode == "floor":
        return np.array([1, 0])
    elif mode == "ceil":
        return np.array([0, 1])


def compare_with_tensorflow(
    device_type, input_shape, dtype, size, data_format, interpolation, align_corners
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def UpsampleJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "input",
                shape=input_shape,
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=True,
            )
            loss = flow.layers.upsample_2d(
                x,
                size=size,
                data_format=data_format,
                interpolation=interpolation,
                align_corners=align_corners,
            )
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0
            ).minimize(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
            return loss

    of_out = UpsampleJob().get()
    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x").astype(np.float32))
        tf_out = tf.keras.layers.UpSampling2D(
            size=size, data_format=channel_pos, interpolation=interpolation
        )(x)
    loss_diff = test_global_storage.Get("loss_diff").astype(np.float32)
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=1e-05, atol=1e-05)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-05, atol=1e-05
    )


def compare_with_numpy(
    device_type, input_shape, dtype, size, data_format, interpolation, align_corners
):
    assert device_type in ["gpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="predict", function_config=func_config)
    def UpsampleJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "input",
                shape=input_shape,
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=False,
            )
            loss = flow.layers.upsample_2d(
                x,
                size=size,
                data_format=data_format,
                interpolation=interpolation,
                align_corners=align_corners,
            )
            flow.watch(x, test_global_storage.Setter("x1"))
            flow.watch(loss, test_global_storage.Setter("loss1"))
            return loss

    of_out = UpsampleJob().get()
    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"
    if align_corners:
        assert interpolation == "bilinear"
        x = test_global_storage.Get("x1")
        if data_format == "NHWC":
            x = np.transpose(x, axes=[0, 3, 1, 2])
        coeffs_dict = {"bilinear": linear_coeffs}
        coeffs = coeffs_dict[interpolation]
        scaler = "align_corners"
        np_out = interpolate_nd(x, coeffs, scale_factors=size, scaler=scaler).astype(
            np.float32
        )
        of_out_np = of_out.numpy()
        if data_format == "NHWC":
            of_out_np = np.transpose(of_out_np, axes=[0, 3, 1, 2])
        assert np.allclose(of_out_np, np_out, rtol=1e-05, atol=1e-05)
    else:
        x = test_global_storage.Get("x1")
        if data_format == "NHWC":
            x = np.transpose(x, axes=[0, 3, 1, 2])
        coeffs_dict = {"bilinear": linear_coeffs, "nearest": nearest_coeffs}
        coeffs = coeffs_dict[interpolation]
        scaler = "pytorch_half_pixel"
        np_out = interpolate_nd(x, coeffs, scale_factors=size, scaler=scaler).astype(
            np.float32
        )
        of_out_np = of_out.numpy()
        if data_format == "NHWC":
            of_out_np = np.transpose(of_out_np, axes=[0, 3, 1, 2])
        assert np.allclose(of_out_np, np_out, rtol=1e-05, atol=1e-05)


@flow.unittest.skip_unless_1n1d()
class TestUpsample(flow.unittest.TestCase):
    def test_upsample(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["input_shape"] = [(2, 11, 12, 13)]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["size"] = [(2, 2), 3, (1, 2)]
        arg_dict["data_format"] = ["NCHW", "NHWC"]
        arg_dict["interpolation"] = ["nearest", "bilinear"]
        arg_dict["align_corners"] = [False]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_upsample_align_corners(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["input_shape"] = [(2, 5, 6, 7)]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["size"] = [(2, 2), 3, (1, 2)]
        arg_dict["data_format"] = ["NCHW", "NHWC"]
        arg_dict["interpolation"] = ["bilinear"]
        arg_dict["align_corners"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_with_numpy(*arg)


if __name__ == "__main__":
    unittest.main()
