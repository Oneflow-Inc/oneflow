import oneflow as flow
import numpy as np
import oneflow.typing as oft
from test_util import GenArgList
import oneflow.typing as oft
import unittest
from collections import OrderedDict

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
    flow.env.init()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    #global function needs float32 as type of argument and return value
    if value_type == flow.float16:
        func_config.default_data_type(flow.float32)
    else:
        func_config.default_data_type(value_type)

    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    func_config.default_logical_view(flow.scope.consistent_view())

    def _compare_diff(blob: oft.Numpy):
        if (np.allclose(grad, blob)) == False:
            print("torch grad:", grad)
            print("oenflow grad:", blob)

    if value_type == flow.float16:
        @flow.global_function(type="train", function_config=func_config)
        def gather_fn(
            params_def: oft.Numpy.Placeholder(input.shape, dtype=flow.float32),
            indices_def: oft.Numpy.Placeholder(index.shape, dtype=index_type),
        ) -> oft.Numpy:
            with flow.scope.placement(device_type, machine_ids):
                x_var = flow.get_variable(
                    "input",
                    shape=input.shape,
                    dtype=flow.float32,
                    initializer=flow.constant_initializer(0),
                )
                x = x_var + params_def
                x_f16 = flow.cast(x, flow.float16)
                y_f16 = flow.dim_gather(x_f16, dim, indices_def)
                x_f32 = flow.cast(x, flow.float32)
                y_f32 = flow.cast(y_f16, flow.float32)
            flow.watch_diff(x_f32, _compare_diff)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(y_f32)
            return y_f32
        return gather_fn
    else:
        @flow.global_function(type="train", function_config=func_config)
        def gather_fn(
            params_def: oft.Numpy.Placeholder(input.shape, dtype=value_type),
            indices_def: oft.Numpy.Placeholder(index.shape, dtype=index_type),
        ) -> oft.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                x_var = flow.get_variable(
                    "input",
                    shape=input.shape,
                    dtype=value_type,
                    initializer=flow.constant_initializer(0),
                )

                x = x_var + params_def
                y = flow.dim_gather(x, dim, indices_def)

            flow.watch_diff(x, _compare_diff)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(y)
            return y

        return gather_fn

def _compare_dim_gather_with_samples(test_case, device_type, sample, value_type, index_type, machine_ids, device_count):
    gather_fn = _make_dim_gather_fn(test_case, 
                    sample["input"].astype(value_type[0]),
                    sample["index"].astype(index_type[0]),
                    sample["dim"],
                    sample["grad"].astype(value_type[0]),
                    device_type,
                    value_type[1],
                    index_type[1],
                    "0:0",
                    1
                    )
    y = gather_fn(sample["input"].astype(value_type[0]), sample["index"].astype(index_type[0]))
    y.astype(value_type[0])

    if value_type == flow.float16:
        test_case.assertTrue(np.allclose(y, sample["output"].astype(np.float32), 1e-3, 1e-3))
    else:
        test_case.assertTrue(np.allclose(y, sample["output"].astype(value_type[0]), 1e-3, 1e-3))

@flow.unittest.skip_unless_1n1d()
class TestDimGather1n1d(flow.unittest.TestCase):
    def test_dim_gather(test_case):
        global g_samples
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["samples"] = []
        arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 1))
        arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 0))
        arg_dict["samples"].append(gen_gather_test_sample((8, 3, 2), (4, 3, 2), 0))
        arg_dict["value_type"] = [(np.float32, flow.float16), (np.float32, flow.float32), (np.float64, flow.float64)]
        arg_dict["index_type"] = [(np.int32, flow.int32)]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_count"] = [1]
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)

@flow.unittest.skip_unless_1n2d()
class TestDimGather1n2d(flow.unittest.TestCase):
    def test_dim_gather(test_case):
        global g_samples
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["samples"] = []
        arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 1))
        arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 0))
        arg_dict["samples"].append(gen_gather_test_sample((8, 3, 2), (4, 3, 2), 0))
        arg_dict["value_type"] = [(np.float32, flow.float16), (np.float32, flow.float32), (np.float64, flow.float64)]
        arg_dict["index_type"] = [(np.int32, flow.int32)]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_count"] = [2]
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)

if __name__ == "__main__":
    unittest.main()
