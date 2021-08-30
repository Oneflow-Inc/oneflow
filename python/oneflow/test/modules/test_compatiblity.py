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
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_alexnet.py", "alexnet", "cuda"
        )

    def test_resnet_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet18", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet34", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet50", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet101", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet152", "cuda"
        )

    def test_resnext_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnext50_32x4d", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnext101_32x8d", "cuda"
        )
    
    def test_wide_resnet_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "wide_resnet50_2", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "wide_resnet101_2", "cuda"
        )

    def test_mnasnet_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet0_5", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet0_75", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet1_0", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet1_3", "cuda"
        )
    
    def test_vgg_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg11", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg13", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg16", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg19", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg11_bn", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg13_bn", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg16_bn", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg19_bn", "cuda"
        )
    
    def test_squeezenet_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_squeezenet.py", "squeezenet1_0", "cuda"
        )
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_squeezenet.py", "squeezenet1_1", "cuda"
        )

if __name__ == "__main__":
    unittest.main()
