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
import time
import tempfile
import multiprocessing

import oneflow as flow
import oneflow.unittest


def _reset_session():
    # Close session to avoid the buffer name duplicate error.
    oneflow.framework.session_context.TryCloseDefaultSession()
    time.sleep(5)
    flow.framework.session_context.NewDefaultSession(flow._oneflow_global_unique_env)


def _with_new_session(fn):
    def new_fn(*args, **kwargs):
        # Avoid Singleton value duplication such as buffer names.
        # saved and loaded graph runtime share the same buffer names(job names).
        print(
            "function ",
            fn.__name__,
            " session reset to avoid Singleton value duplication ...",
        )
        _reset_session()
        out = fn(*args, **kwargs)
        _reset_session()
        return out

    return new_fn


def _test_linear_multi_graph_share(test_case, device, with_reshape):
    linear = flow.nn.Linear(3, 8, False)
    linear = linear.to(device)
    np_weight = np.ones((3, 8)).astype(np.float32)
    np_weight.fill(2.3)
    flow.nn.init.constant_(linear.weight, 2.3)

    class LinearReshapeModule(flow.nn.Module):
        def __init__(self, lin, with_r):
            super().__init__()
            self.linear = lin
            self.with_reshape = with_r

        def forward(self, x):
            y = self.linear(x)
            if with_reshape:
                assert len(y.shape) == 2
                return flow.reshape(y, (y.shape[1], y.shape[0]))
            else:
                return y

    linear_reshape = LinearReshapeModule(linear, with_reshape)

    class LinearGraph(flow.nn.Graph):
        @flow.nn.Graph.with_dynamic_input_shape(size=4)
        def __init__(self, lin, with_r):
            super().__init__()
            self.my_linear = LinearReshapeModule(lin, with_r)

        def build(self, x):
            return self.my_linear(x)

    linear_g = LinearGraph(linear, with_reshape)
    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    x = flow.tensor(input_arr, device=device)
    of_lazy_out = linear_g(x)
    of_eager_out = linear_reshape(x)
    test_case.assertTrue(np.array_equal(of_lazy_out.numpy(), of_eager_out.numpy()))

    input_arr1 = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
        ],
        dtype=np.float32,
    )
    x1 = flow.tensor(input_arr1, device=device)
    of_lazy_out1 = linear_g(x1)
    of_eager_out1 = linear_reshape(x1)
    test_case.assertTrue(np.array_equal(of_lazy_out1.numpy(), of_eager_out1.numpy()))

    input_arr2 = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
        ],
        dtype=np.float32,
    )
    x2 = flow.tensor(input_arr2, device=device)
    of_lazy_out2 = linear_g(x2)
    of_eager_out2 = linear_reshape(x2)
    test_case.assertTrue(np.array_equal(of_lazy_out2.numpy(), of_eager_out2.numpy()))

    of_lazy_out2 = linear_g(x2)
    of_eager_out2 = linear_reshape(x2)
    test_case.assertTrue(np.array_equal(of_lazy_out2.numpy(), of_eager_out2.numpy()))


def _get_state_dict_tensor_size(sd):
    from oneflow.framework.args_tree import ArgsTree

    def _get_tensor_mem(input):
        # if input.dim() == 0:
        #     return 2
        cnt_size = input.element_size() * flow.numel(input)
        return cnt_size

    args_tree = ArgsTree(sd, False)

    size = 0
    for arg in args_tree.iter_nodes():
        if isinstance(arg, flow.Tensor):
            size += _get_tensor_mem(arg)
        else:
            continue
    return size


@_with_new_session
def _test_linear_multi_graph_save(return_dict, device, with_reshape, with_eager):
    linear = flow.nn.Linear(3, 8, False)
    linear = linear.to(device)
    np_weight = np.ones((3, 8)).astype(np.float32)
    np_weight.fill(2.3)
    flow.nn.init.constant_(linear.weight, 2.3)

    class LinearReshapeModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            y = self.linear(x)
            if with_reshape:
                assert len(y.shape) == 2
                return flow.reshape(y, (y.shape[1], y.shape[0]))
            else:
                return y

    linear_reshape = LinearReshapeModule()

    class LinearGraph(flow.nn.Graph):
        @flow.nn.Graph.with_dynamic_input_shape(size=3)
        def __init__(self):
            super().__init__(enable_get_runtime_state_dict=True)
            self.my_linear = linear_reshape

        def build(self, x):
            return self.my_linear(x)

    linear_g = LinearGraph()

    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    x = flow.tensor(input_arr, device=device)
    of_lazy_out = linear_g(x)
    of_eager_out = linear_reshape(x)
    test_case0 = np.array_equal(of_lazy_out.numpy(), of_eager_out.numpy())
    return_dict["save0"] = test_case0

    input_arr1 = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
        ],
        dtype=np.float32,
    )
    x1 = flow.tensor(input_arr1, device=device)
    of_lazy_out1 = linear_g(x1)
    of_eager_out1 = linear_reshape(x1)
    test_case1 = np.array_equal(of_lazy_out1.numpy(), of_eager_out1.numpy())
    return_dict["save1"] = test_case1

    input_arr2 = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
        ],
        dtype=np.float32,
    )
    x2 = flow.tensor(input_arr2, device=device)
    of_lazy_out2 = linear_g(x2)
    of_eager_out2 = linear_reshape(x2)
    test_case2 = np.array_equal(of_lazy_out2.numpy(), of_eager_out2.numpy())
    return_dict["save2"] = test_case2

    input_arr3 = np.array([[-0.94630778, -0.83378579, -0.87060891],], dtype=np.float32,)
    x3 = flow.tensor(input_arr3, device=device)
    of_lazy_out3 = linear_g(x3)
    of_eager_out3 = linear_reshape(x3)
    test_case3 = np.array_equal(of_lazy_out3.numpy(), of_eager_out3.numpy())
    return_dict["save3"] = test_case3

    of_lazy_out1 = linear_g(x1)
    test_case1 = np.array_equal(of_lazy_out1.numpy(), of_eager_out1.numpy())
    return_dict["save4"] = test_case1

    state_dict = linear_g.runtime_state_dict(with_eager=with_eager)
    print("====> saved graphs", state_dict.keys())
    return state_dict


@_with_new_session
def _test_linear_multi_graph_load(return_dict, device, with_reshape, state_dict):
    linear = flow.nn.Linear(3, 8, False)
    linear = linear.to(device)
    np_weight = np.ones((3, 8)).astype(np.float32)
    np_weight.fill(2.3)
    flow.nn.init.constant_(linear.weight, 2.3)

    class LinearReshapeModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            y = self.linear(x)
            if with_reshape:
                assert len(y.shape) == 2
                return flow.reshape(y, (y.shape[1], y.shape[0]))
            else:
                return y

    linear_reshape = LinearReshapeModule()

    class LinearGraph(flow.nn.Graph):
        @flow.nn.Graph.with_dynamic_input_shape(size=2)
        def __init__(self):
            super().__init__()
            self.my_linear = linear_reshape

        def build(self, x):
            return self.my_linear(x)

    linear_g = LinearGraph()
    print("====> load")
    linear_g.load_runtime_state_dict(state_dict)
    print("====> load finish")

    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    x = flow.tensor(input_arr, device=device)
    of_lazy_out = linear_g(x)
    of_eager_out = linear_reshape(x)
    test_case0 = np.array_equal(of_lazy_out.numpy(), of_eager_out.numpy())
    return_dict["load0"] = test_case0

    input_arr1 = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
        ],
        dtype=np.float32,
    )
    x1 = flow.tensor(input_arr1, device=device)
    of_lazy_out1 = linear_g(x1)
    of_eager_out1 = linear_reshape(x1)
    test_case1 = np.array_equal(of_lazy_out1.numpy(), of_eager_out1.numpy())
    return_dict["load1"] = test_case1

    # TODO(strint): shared from a load graph.


def _graph_save(return_dict, filename, with_eager):
    state_dict = _test_linear_multi_graph_save(
        return_dict, flow.device("cuda"), True, with_eager
    )
    print(
        f"state_dict(with_eager={with_eager}) tensors size ",
        _get_state_dict_tensor_size(state_dict),
    )
    flow.save(state_dict, filename)


def _graph_load(return_dict, filename):
    state_dict_loaded = flow.load(filename)
    # load with nn.Graph
    _test_linear_multi_graph_load(
        return_dict, flow.device("cuda"), True, state_dict_loaded,
    )


def _test_linear_multi_graph_save_load_gpu(test_case, with_eager):
    # A graph runtime state dict
    with tempfile.NamedTemporaryFile() as f:
        # Save a graph
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        save_p = multiprocessing.get_context("spawn").Process(
            target=_graph_save, args=(return_dict, f.name, with_eager)
        )
        save_p.start()
        save_p.join()

        # Resume a graph from a graph runtime state dict
        load_p = multiprocessing.get_context("spawn").Process(
            target=_graph_load, args=(return_dict, f.name)
        )
        load_p.start()
        load_p.join()

        # test_case can't be passed into sub process, so we check with return_dict.
        # Reference: https://stackoverflow.com/questions/52225003/writing-to-multiple-files-using-multiprocessing-error-typeerror-cannot-seria
        for (key, check_value) in return_dict.items():
            test_case.assertTrue(check_value, key + " failed.")


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearMultiGraph(oneflow.unittest.TestCase):
    def test_linear_multi_graph_share_gpu(test_case):
        _test_linear_multi_graph_share(test_case, flow.device("cuda"), False)

    def test_linear_reshape_multi_graph_share_gpu(test_case):
        _test_linear_multi_graph_share(test_case, flow.device("cuda"), True)

    def test_linear_multi_graph_save_load_gpu_with_share(test_case):
        _test_linear_multi_graph_save_load_gpu(test_case, True)

    def test_linear_multi_graph_save_load_gpu_with_share_without_eager(test_case):
        _test_linear_multi_graph_save_load_gpu(test_case, False)


if __name__ == "__main__":
    unittest.main()
