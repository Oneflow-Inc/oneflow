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
import unittest
from collections import OrderedDict

import numpy as np
import math

import oneflow as flow
from oneflow.test_utils.test_util import GenArgList


input_np = np.array(
    [
        [
            [
                [
                    0.33840093,
                    1.1469249,
                    1.0410756,
                    -0.8350606,
                    -1.782742,
                    -0.00350855,
                    -0.45829752,
                    -1.0764053,
                ],
                [
                    -0.4169678,
                    -0.07322863,
                    1.5186151,
                    1.3238515,
                    -0.3002863,
                    0.90660757,
                    -0.2955834,
                    1.5069526,
                ],
                [
                    0.3829125,
                    1.0149552,
                    -0.5808607,
                    -0.4644214,
                    1.2142111,
                    0.668561,
                    1.0866925,
                    0.16446872,
                ],
                [
                    0.14043295,
                    -0.55108964,
                    -0.8154048,
                    1.1554539,
                    2.421505,
                    -0.54017824,
                    0.32610297,
                    -1.0632077,
                ],
                [
                    -0.6218423,
                    0.6000421,
                    0.3742695,
                    0.11130165,
                    0.9991065,
                    -0.28596586,
                    -0.05164787,
                    0.07725058,
                ],
                [
                    0.6141537,
                    0.2919493,
                    0.2101646,
                    -0.16639,
                    1.145933,
                    0.08825321,
                    0.9865119,
                    0.47285828,
                ],
                [
                    -1.5073836,
                    -0.8056736,
                    -0.7402776,
                    -0.9932287,
                    0.74761075,
                    -0.46474454,
                    -0.22881153,
                    0.6082243,
                ],
                [
                    0.8328902,
                    0.17223845,
                    0.48917648,
                    -1.6264182,
                    0.248678,
                    -1.2603166,
                    1.2644174,
                    0.06434552,
                ],
            ]
        ],
        [
            [
                [
                    0.6627289,
                    0.68173873,
                    0.17659399,
                    0.17474514,
                    0.72995424,
                    -0.47240442,
                    0.27204773,
                    -0.5277862,
                ],
                [
                    0.23609516,
                    0.9604236,
                    0.78075147,
                    0.26125216,
                    0.72746485,
                    0.04412199,
                    0.04948105,
                    -0.08477508,
                ],
                [
                    0.8646437,
                    -0.20755729,
                    1.0184883,
                    0.06346282,
                    -0.18039183,
                    0.56243396,
                    -0.07350786,
                    -1.8523406,
                ],
                [
                    -0.2267861,
                    -1.6466936,
                    2.1746075,
                    -1.2284307,
                    0.74488103,
                    -0.13243976,
                    -0.9046582,
                    -2.2992454,
                ],
                [
                    -0.56131303,
                    -0.17723852,
                    -0.6063047,
                    2.4105318,
                    0.96672636,
                    -1.8386889,
                    1.1021106,
                    -0.65429336,
                ],
                [
                    2.0618255,
                    -0.86972237,
                    -0.59159493,
                    0.9894253,
                    -0.26607743,
                    -0.395585,
                    -0.44035113,
                    -0.663197,
                ],
                [
                    -0.02398485,
                    -0.04574186,
                    -0.43163615,
                    -0.42599657,
                    -2.751177,
                    -0.35520887,
                    -0.413676,
                    2.0098279,
                ],
                [
                    1.5619192,
                    -2.4961088,
                    0.08771367,
                    -2.289146,
                    1.0729461,
                    0.7120767,
                    -0.09780294,
                    -1.6628668,
                ],
            ]
        ],
    ]
)

rois_np = np.array(
    [
        [1.0, 2.0, 1.0324688, 2.5, 3.90168],
        [1.0, 2.5, 2.8329468, 3.5, 3.2008305],
        [0.0, 1.0, 1.6188955, 2.0, 0.99051666],
        [1.0, 1.0, 1.843338, 1.0, 3.9240131],
        [1.0, 2.0, 2.798994, 3.5, 1.2012959],
        [0.0, 0.5, 2.7753997, 3.0, 0.8280029],
        [1.0, 0.5, 2.167975, 2.0, 2.067833],
        [0.0, 0.5, 2.6843219, 2.0, 3.9924717],
        [0.0, 2.0, 2.8996983, 3.5, 2.356554],
        [0.0, 1.5, 0.34730053, 3.0, 2.8540745],
        [0.0, 0.0, 2.096885, 0.5, 3.357812],
        [0.0, 1.5, 0.10133362, 3.0, 0.18236923],
        [1.0, 1.0, 1.609498, 1.5, 3.8893862],
        [0.0, 1.5, 0.03415012, 1.5, 1.2880297],
        [0.0, 0.5, 3.9403543, 2.0, 3.8870106],
        [0.0, 0.0, 3.7515945, 3.5, 0.5866394],
        [1.0, 1.5, 1.7729645, 2.0, 1.2372265],
        [1.0, 0.0, 1.5092888, 2.0, 3.1585617],
        [1.0, 0.0, 2.9033833, 1.5, 1.659832],
        [1.0, 0.5, 1.9115062, 3.0, 1.066021],
        [0.0, 1.5, 3.185645, 2.0, 0.20558739],
        [1.0, 2.0, 0.3081894, 2.5, 2.4888725],
        [0.0, 0.5, 3.5662794, 3.5, 2.8792458],
        [1.0, 0.5, 2.556768, 2.5, 2.1553097],
        [0.0, 1.0, 1.397994, 3.5, 0.77407074],
        [0.0, 0.5, 3.1722808, 3.5, 2.5378036],
        [0.0, 0.5, 0.11013985, 3.5, 0.8963146],
        [0.0, 2.0, 1.1824799, 2.0, 3.2211132],
        [1.0, 0.0, 3.9227288, 2.0, 2.0894089],
        [0.0, 1.0, 0.79490566, 1.5, 3.4291687],
    ]
)

input_grad_np = np.array(
    [
        [
            [
                [
                    0.2517704,
                    1.7398968,
                    8.248332,
                    16.302334,
                    11.048147,
                    10.059495,
                    2.800579,
                    0.24844748,
                ],
                [
                    0.790752,
                    3.154358,
                    13.0182705,
                    15.519342,
                    7.0133696,
                    6.28652,
                    3.9538488,
                    0.51601994,
                ],
                [
                    0.7077478,
                    3.6854784,
                    19.228241,
                    22.597464,
                    10.153106,
                    6.2180595,
                    3.5736852,
                    0.44621366,
                ],
                [
                    1.1430397,
                    2.6666558,
                    8.699481,
                    12.510508,
                    7.6093874,
                    3.3150473,
                    1.0373969,
                    0.08225401,
                ],
                [
                    7.372374,
                    3.458156,
                    6.5517087,
                    10.535179,
                    9.493686,
                    5.800008,
                    3.2196481,
                    0.3790145,
                ],
                [
                    9.979998,
                    7.723156,
                    11.384828,
                    15.13672,
                    14.71994,
                    11.550301,
                    8.666647,
                    1.1556869,
                ],
                [
                    7.4674473,
                    7.990606,
                    11.032139,
                    10.031732,
                    6.5969977,
                    5.1203485,
                    4.1267443,
                    0.57233953,
                ],
                [
                    1.9118737,
                    10.9567,
                    12.461995,
                    10.991727,
                    2.2403586,
                    0.9002282,
                    0.74645257,
                    0.1000254,
                ],
            ]
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.2796778, 1.6780672, 0.2796781, 0.0, 0.0],
                [
                    0.02485762,
                    0.17400333,
                    0.19886094,
                    0.94998413,
                    4.7056007,
                    0.9251272,
                    0.02485762,
                    0.0,
                ],
                [
                    0.54076296,
                    2.3330488,
                    4.2377095,
                    13.100019,
                    12.285746,
                    4.7681584,
                    1.6636131,
                    0.18542966,
                ],
                [
                    4.555413,
                    9.538326,
                    14.063398,
                    17.882318,
                    14.635002,
                    5.9126663,
                    2.6039343,
                    0.3144545,
                ],
                [
                    7.877132,
                    19.767809,
                    24.037426,
                    15.584505,
                    14.542083,
                    4.4302306,
                    2.3387682,
                    0.3145125,
                ],
                [
                    6.3498077,
                    11.157468,
                    14.465272,
                    6.4254785,
                    7.471047,
                    7.448948,
                    6.4972777,
                    0.88493776,
                ],
                [
                    2.473032,
                    6.144208,
                    9.52839,
                    3.2779343,
                    4.3061023,
                    6.409383,
                    5.87155,
                    0.80066556,
                ],
                [
                    1.4289956,
                    4.5101476,
                    7.2189507,
                    2.2500885,
                    2.8763475,
                    0.45081174,
                    0.0,
                    0.0,
                ],
            ]
        ],
    ]
)


def bilinear_interpolate(data, y, x, snap_border=False):
    height, width = data.shape

    if snap_border:
        if -1 < y <= 0:
            y = 0
        elif height - 1 <= y < height:
            y = height - 1

        if -1 < x <= 0:
            x = 0
        elif width - 1 <= x < width:
            x = width - 1

    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = y_low + 1
    x_high = x_low + 1

    wy_h = y - y_low
    wx_h = x - x_low
    wy_l = 1 - wy_h
    wx_l = 1 - wx_h

    val = 0
    for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
            if 0 <= yp < height and 0 <= xp < width:
                val += wx * wy * data[yp, xp]
    return val


def roi_align_np(
    in_data,
    rois,
    pool_h,
    pool_w,
    spatial_scale=1,
    sampling_ratio=-1,
    aligned=False,
    dtype=np.float32,
):
    n_channels = in_data.shape[1]
    out_data = np.zeros((rois.shape[0], n_channels, pool_h, pool_w), dtype=dtype)

    offset = 0.5 if aligned else 0.0

    for r, roi in enumerate(rois):
        batch_idx = int(roi[0])
        j_begin, i_begin, j_end, i_end = (
            x.item() * spatial_scale - offset for x in roi[1:]
        )

        roi_h = i_end - i_begin
        roi_w = j_end - j_begin
        bin_h = roi_h / pool_h
        bin_w = roi_w / pool_w

        for i in range(0, pool_h):
            start_h = i_begin + i * bin_h
            grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
            for j in range(0, pool_w):
                start_w = j_begin + j * bin_w
                grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))

                for channel in range(0, n_channels):

                    val = 0
                    for iy in range(0, grid_h):
                        y = start_h + (iy + 0.5) * bin_h / grid_h
                        for ix in range(0, grid_w):
                            x = start_w + (ix + 0.5) * bin_w / grid_w
                            val += bilinear_interpolate(
                                in_data[batch_idx, channel, :, :],
                                y,
                                x,
                                snap_border=True,
                            )
                    val /= grid_h * grid_w

                    out_data[r, channel, i, j] = val
    return out_data


def _test_roi_align(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 3, 64, 64), dtype=flow.float32, device=flow.device(device)
    )

    random_img_idx = np.random.randint(low=0, high=2, size=(200, 1))
    random_box_idx = np.random.uniform(low=0, high=64 * 64, size=(200, 2)).astype(
        np.float32
    )

    def get_h_w(idx1, idx2):
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        h1 = idx1 // 64
        w1 = idx1 % 64
        h2 = idx2 // 64
        w2 = idx2 % 64
        return [x / 2 for x in [h1, w1, h2, w2]]

    zipped = zip(random_box_idx[:, 0], random_box_idx[:, 1])
    concated = [get_h_w(idx1, idx2) for (idx1, idx2) in zipped]
    concated = np.array(concated)
    rois = flow.tensor(
        np.hstack((random_img_idx, concated)),
        dtype=flow.float32,
        device=flow.device(device),
    )

    of_out = flow.roi_align(input, rois, 2.0, 14, 14, 2, True)
    np_out = roi_align_np(input.numpy(), rois.numpy(), 14, 14, 2.0, 2, True)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-4, atol=1e-4))


def _test_roi_align_backward(test_case, device):
    input = flow.tensor(
        input_np, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    rois = flow.tensor(rois_np, dtype=flow.float32, device=flow.device(device))
    of_out = flow.roi_align(input, rois, 2.0, 5, 5, 2, True)
    of_out.sum().backward()
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), input_grad_np, rtol=1e-5, atol=1e-5)
    )


@flow.unittest.skip_unless_1n1d()
class TestRoIAlign(flow.unittest.TestCase):
    def test_roi_align(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_roi_align, _test_roi_align_backward]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
