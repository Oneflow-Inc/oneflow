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
import numpy as np
import oneflow as flow


def test_underline_gather():
    x = flow.tensor(np.random.randn(10, 128), device="cpu", dtype=flow.float32)
    index = flow.tensor(
        np.random.randint(low=1, high=3, size=(1, 3)), device="cpu", dtype=flow.int32
    )
    cpu_out_numpy = flow._C.gather(x, index, axis=0).numpy()
    x = x.to("mlu")
    index = index.to("mlu")
    mlu_out_numpy = flow._C.gather(x, index, axis=0).numpy()
    assert np.allclose(cpu_out_numpy, mlu_out_numpy, 1e-4, 1e-4)
