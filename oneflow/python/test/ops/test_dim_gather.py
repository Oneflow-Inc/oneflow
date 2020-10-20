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
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList
import oneflow.typing as oft
import unittest


def gen_gather_test_sample(input_shape, index_shape, dim):
    def _flatten_array(input_array):
        output_array = list()
        for x in np.nditer(input_array):
            output_array.append(x.tolist())
        return output_array

    def _offset2coordinate(offset, tensor_shape):
        coordinate = []
        tmp = offset
        for i in range(len(tensor_shape) - 1, -1, -1):
            axis_size = tensor_shape[i]
            coor = tmp % axis_size
            coordinate.insert(0, int(coor))
            tmp = (tmp - coor) / axis_size

        return coordinate

    def _coordinate2offset(coordinate, tensor_shape):
        if len(coordinate) != len(tensor_shape):
            raise "wrong coordinate or shape"
        offset = 0
        for i, coor in enumerate(coordinate):
            size_at_axis = coor
            for j in range(i + 1, len(tensor_shape)):
                size_at_axis *= tensor_shape[j]

            offset += size_at_axis
        return offset

    def _torch_gather(input, dim, index):
        output = np.zeros(index.shape)
        output = _flatten_array(output)
        input1d = _flatten_array(input)
        for idxoffset in range(0, index.size):
            index1d = _flatten_array(index)
            x = index1d[idxoffset]
            coor_index = _offset2coordinate(idxoffset, index.shape)
            coor_index[dim] = x

            input_offset = _coordinate2offset(coor_index, input.shape)
            output[idxoffset] = input1d[input_offset]
        ret = np.resize(np.array(output), index.shape)
        return np.array(ret)

    def _torch_scatter_add(src, dim, index, outshape):
        output = np.zeros(outshape)
        output1d = _flatten_array(output)
        index1d = _flatten_array(index)
        src1d = _flatten_array(src)
        for srcidx in range(0, src.size):
            outcoord = _offset2coordinate(srcidx, src.shape)
            outcoord[dim] = index1d[srcidx]

            output_offset = _coordinate2offset(outcoord, outshape)
            output1d[output_offset] += src1d[srcidx]

        ret = np.resize(np.array(output1d), outshape)
        return ret

    input = np.random.random(input_shape)
    index = np.random.randint(0, input_shape[dim], index_shape)
    output = _torch_gather(input, dim, index)
    grad = _torch_scatter_add(np.ones_like(output), dim, index, input_shape)

    ret = {"input": input, "index": index, "dim": dim, "output": output, "grad": grad}
    return ret


def _make_dim_gather_fn(
    test_case,
    input,
    index,
    dim,
    grad,
    device_type,
    value_type,
    index_type,
    machine_ids,
    device_counts,
):
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(1)
        machine_ids = "0:0"
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_data_type(value_type)
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    func_config.default_logical_view(flow.scope.consistent_view())

    def _compare_diff(blob: oft.Numpy):
        test_case.assertTrue(np.array_equal(grad, blob))

    def do_gather(x_blob, i_blob):
        with flow.scope.placement(device_type, machine_ids):
            with flow.scope.placement(device_type, "0:0-0"):
                x = flow.get_variable(
                    "input",
                    shape=input.shape,
                    dtype=value_type,
                    initializer=flow.constant_initializer(0),
                )
                x = flow.cast_to_current_logical_view(x)
                x_blob = flow.cast_to_current_logical_view(x_blob)
                x = x + x_blob

            y = flow.dim_gather(x, dim, i_blob)
            flow.watch_diff(x, _compare_diff)

        with flow.scope.placement(device_type, "0:0-0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(y)
        return y

    @flow.global_function(type="train", function_config=func_config)
    def gather_fn(
        params_def: oft.Numpy.Placeholder(input.shape, dtype=value_type),
        indices_def: oft.Numpy.Placeholder(index.shape, dtype=index_type),
    ):
        return do_gather(params_def, indices_def)

    return gather_fn


def _compare_dim_gather_with_samples(
    test_case, device_type, sample, value_type, index_type, machine_ids, device_counts
):
    input = sample["input"].astype(value_type[0])
    index = sample["index"].astype(index_type[0])
    out = sample["output"].astype(np.float32)
    dim = sample["dim"]
    grad = sample["grad"].astype(value_type[0])

    params, indices = input, index
    gather_fn = _make_dim_gather_fn(
        test_case,
        params,
        indices,
        dim,
        grad,
        device_type,
        value_type[1],
        index_type[1],
        machine_ids,
        device_counts,
    )

    of_y = gather_fn(params, indices).get().numpy()

    test_case.assertTrue(np.allclose(out, of_y))


@flow.unittest.skip_unless_1n1d()
class TestDimGather1n1d(flow.unittest.TestCase):
    def test_dim_gather_cpu(test_case):
        global g_samples
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["samples"] = []
        arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 1))
        arg_dict["samples"].append(gen_gather_test_sample((8, 4, 2), (4, 4, 2), 0))
        arg_dict["value_type"] = [(np.float32, flow.float32), (np.double, flow.double)]
        arg_dict["index_type"] = [(np.int32, flow.int32), (np.int64, flow.int64)]
        arg_dict["machine_ids"] = ["0:0-0"]
        arg_dict["device_count"] = [1]
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)

    def test_dim_gather_gpu(test_case):
        global g_samples
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["samples"] = []
        arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 1))
        arg_dict["samples"].append(gen_gather_test_sample((8, 4, 2), (4, 4, 2), 0))
        arg_dict["value_type"] = [(np.float32, flow.float32), (np.double, flow.double)]
        arg_dict["index_type"] = [(np.int32, flow.int32), (np.int64, flow.int64)]
        arg_dict["machine_ids"] = ["0:0-0"]
        arg_dict["device_count"] = [1]
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestDimGather1n2dConsistent(flow.unittest.TestCase):
    def test_dim_gather_2cards(test_case):
        flow.clear_default_session()
        global g_samples
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["samples"] = []
        arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 1))
        arg_dict["samples"].append(gen_gather_test_sample((8, 4, 2), (4, 4, 2), 0))
        arg_dict["value_type"] = [(np.float32, flow.float32), (np.double, flow.double)]
        arg_dict["index_type"] = [(np.int32, flow.int32), (np.int64, flow.int64)]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_count"] = [2]
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
