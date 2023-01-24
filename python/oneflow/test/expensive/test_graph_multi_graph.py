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


@_with_new_session
def _test_linear_multi_graph_share(test_case, device, with_reshape):
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
        def __init__(self):
            super().__init__()
            self.my_linear = linear_reshape

        def build(self, x):
            return self.my_linear(x)

    linear_g = LinearGraph()
    linear_g.enable_shared()
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
    print("graph 0 out ", of_lazy_out)

    linear_g1 = LinearGraph()
    linear_g1.share_from(linear_g)
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
    of_lazy_out1 = linear_g1(x1)
    print("graph 1 out ", of_lazy_out1)
    of_eager_out1 = linear_reshape(x1)
    test_case.assertTrue(np.array_equal(of_lazy_out1.numpy(), of_eager_out1.numpy()))

    linear_g2 = LinearGraph()
    linear_g2.share_from(linear_g)
    input_arr2 = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
        ],
        dtype=np.float32,
    )
    x2 = flow.tensor(input_arr2, device=device)
    of_lazy_out2 = linear_g2(x2)
    print(" graph 2 out ", of_lazy_out2)
    of_eager_out2 = linear_reshape(x2)
    test_case.assertTrue(np.array_equal(of_lazy_out2.numpy(), of_eager_out2.numpy()))


@_with_new_session
def _test_linear_multi_graph_save(test_case, device, with_reshape):
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
        def __init__(self):
            super().__init__()
            self.my_linear = linear_reshape

        def build(self, x):
            return self.my_linear(x)

    linear_g = LinearGraph()
    linear_g.enable_save_runtime_state_dict()
    linear_g.enable_shared()

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
    print("graph 0 out ", of_lazy_out)

    linear_g1 = LinearGraph()
    linear_g1.enable_save_runtime_state_dict()
    linear_g1.share_from(linear_g)
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
    of_lazy_out1 = linear_g1(x1)
    print("graph 1 out ", of_lazy_out1)
    of_eager_out1 = linear_reshape(x1)
    test_case.assertTrue(np.array_equal(of_lazy_out1.numpy(), of_eager_out1.numpy()))

    state_dict_list = []
    state_dict0 = linear_g.runtime_state_dict()
    state_dict_list.append(state_dict0)
    state_dict1 = linear_g1.runtime_state_dict()
    state_dict_list.append(state_dict1)

    return state_dict_list


@_with_new_session
def _test_linear_multi_graph_load(test_case, device, with_reshape, state_dict_list):
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

    linear_g = flow.nn.Graph()
    linear_g.enable_shared()
    linear_g.load_runtime_state_dict(state_dict_list[0])

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
    print("graph 0 out ", of_lazy_out)

    linear_g1 = flow.nn.Graph()
    linear_g1.share_from(linear_g)
    linear_g1.load_runtime_state_dict(state_dict_list[1])
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
    of_lazy_out1 = linear_g1(x1)
    print("graph 1 out ", of_lazy_out1)
    of_eager_out1 = linear_reshape(x1)
    test_case.assertTrue(np.array_equal(of_lazy_out1.numpy(), of_eager_out1.numpy()))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearMultiGraph(oneflow.unittest.TestCase):
    def test_linear_multi_graph_share_gpu(test_case):
        _test_linear_multi_graph_share(test_case, flow.device("cuda"), False)

    def test_linear_reshape_multi_graph_share_gpu(test_case):
        _test_linear_multi_graph_share(test_case, flow.device("cuda"), True)

    def test_linear_multi_graph_save_load_gpu(test_case):
        # A graph runtime state dict
        state_dict_list = _test_linear_multi_graph_save(
            test_case, flow.device("cuda"), True
        )

        # print("runtime state dict list", state_dict_list)
        with tempfile.TemporaryDirectory() as save_dir:
            flow.save(state_dict_list, save_dir)
            state_dict_list_loaded = flow.load(save_dir, map_location="cuda")

        # Resume a graph from a graph runtime state dict
        _test_linear_multi_graph_load(
            test_case, flow.device("cuda"), True, state_dict_list_loaded
        )


if __name__ == "__main__":
    unittest.main()
