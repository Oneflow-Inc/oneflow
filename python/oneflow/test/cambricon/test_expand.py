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


def test_expand():
    shape_pairs = zip(
        [(1, 4, 1, 32), (2, 4, 1, 32), (1, 6, 5, 3),],
        [(2, 1, 2, 4, 2, 32), (2, 4, 2, 32), (4, -1, 5, 3)],
    )
    dtypes = [flow.float32, flow.int]
    for shapes, dtype in itertools.product(shape_pairs, dtypes):
        original_shape, expand_shape = shapes
        x = flow.tensor(np.random.randn(*original_shape), device="cpu", dtype=dtype)
        cpu_out_numpy = x.expand(*expand_shape).numpy()

        x = x.to("mlu")
        mlu_out_numpy = x.expand(*expand_shape).numpy()
        assert np.allclose(cpu_out_numpy, mlu_out_numpy, 1e-4, 1e-4)
