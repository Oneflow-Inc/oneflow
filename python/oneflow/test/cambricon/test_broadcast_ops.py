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
import numpy as np
import oneflow as flow


def _test_broadcast_forward(op, shape1, shape2, dtype):
    assert len(shape1) == len(shape2)
    x = flow.tensor(np.random.randn(*shape1), device="cpu", dtype=dtype)
    y = flow.tensor(np.random.randn(*shape2), device="cpu", dtype=dtype)
    cpu_out_numpy = op(x, y).numpy()
    x = x.to("mlu")
    y = y.to("mlu")
    mlu_out_numpy = op(x, y).numpy()
    assert np.allclose(cpu_out_numpy, mlu_out_numpy, 1e-4, 1e-4)


def test_broadcast_add_forward():
    shape_pairs = zip(
        [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)], [(1,), (2, 1), (2, 1, 4), (2, 1, 1, 5)]
    )
    default_dtype_list = [flow.float32, flow.float16]
    dtypes = {
        flow.add: [flow.int8, flow.uint8, flow.int32] + default_dtype_list,
        flow.div: default_dtype_list,
        flow.mul: default_dtype_list,
    }
    for op, shapes in itertools.product(dtypes.keys(), shape_pairs):
        for dtype in dtypes[op]:
            _test_broadcast_forward(op, *shapes, dtype)
