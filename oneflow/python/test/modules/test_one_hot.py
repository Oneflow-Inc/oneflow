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
import oneflow.experimental as flow
import oneflow.experimental.nn as nn
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft


def _test_one_hot(test_case, device):
    input = np.array([0, 3, 1, 2]).astype(np.int32)
    np_out = np.array(
        [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],]
    )
    output = nn.Onehot(input, 5, -1, flow.int32)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))


if __name__ == "__main__":
    unittest.main()
