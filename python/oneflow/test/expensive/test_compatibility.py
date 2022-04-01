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
from oneflow.test_utils.oneflow_pytorch_compatibility import *


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestApiCompatibility(flow.unittest.TestCase):
    def test_alexnet_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_alexnet.py", "alexnet", "cuda", 16, 224
        )

    def test_resnet50_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet50", "cuda", 16, 224
        )

    def test_convmixer_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_convmixer.py", "convmixer_768_32_relu", "cuda", 4, 224
        )

    def test_densenet_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_densenet.py", "densenet121", "cuda", 8, 224
        )

    def test_ghostnet_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_ghostnet.py", "ghost_net", "cuda", 16, 224
        )

    def test_googlenet_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_googlenet.py", "googlenet", "cuda", 8, 224
        )

    def test_inception_v3_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_inception_v3.py", "inception_v3", "cuda", 4, 299
        )

    def test_mnasnet_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet1_0", "cuda", 16, 224
        )

    def test_rexnet_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_rexnet.py", "rexnetv1_1_0", "cuda", 16, 224
        )

    def test_rexnetv1_lite_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_rexnetv1_lite.py", "rexnet_lite_1_0", "cuda", 16, 224
        )

    def test_res2net_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_res2net.py", "res2net50", "cuda", 16, 224
        )

    def test_shufflenetv2_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_shufflenetv2.py", "shufflenet_v2_x2_0", "cuda", 16, 224
        )

    def test_squeezenet_compatibility(test_case):
        do_test_train_loss_oneflow_pytorch(
            test_case, "pytorch_squeezenet.py", "squeezenet1_1", "cuda", 16, 224
        )


if __name__ == "__main__":
    unittest.main()
