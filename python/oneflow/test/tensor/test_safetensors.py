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
import tempfile
import oneflow as flow
import oneflow.unittest
import oneflow.mock_torch as mock


tensors = {
    "weight1": flow.zeros((1024, 1024)),
    "weight2": flow.ones((1024, 1024)),
    "weight3": flow.rand((1024, 1024)),
    "weight4": flow.eye(1024),
}


def _test_save_safetensors(save_path):
    with mock.enable():
        from safetensors.torch import save_file

        save_file(tensors, save_path)


def _test_load_safetensors(load_path):
    with mock.enable():
        from safetensors import safe_open

        tensors_load = {}
        with safe_open(load_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors_load[key] = f.get_tensor(key)
        return tensors_load


class TestSafetensors(flow.unittest.TestCase):
    def test_safetensors(test_case):
        with tempfile.TemporaryDirectory() as f0:
            _test_save_safetensors(os.path.join(f0, "model.safetensors"))
            tensors_load = _test_load_safetensors(os.path.join(f0, "model.safetensors"))
        for key in tensors.keys():
            test_case.assertTrue((tensors[key] == tensors_load[key]).all())


if __name__ == "__main__":
    unittest.main()
