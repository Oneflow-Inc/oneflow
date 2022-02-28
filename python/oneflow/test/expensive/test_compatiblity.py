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
from oneflow.test_utils.oneflow_pytorch_compatiblity import *


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestApiCompatiblity(flow.unittest.TestCase):
    def test_alexnet_compatiblity(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_alexnet.py", "alexnet", "cuda"
        )

    def test_resnet50_compatiblity(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet50", "cuda"
        )


if __name__ == "__main__":
    unittest.main()
