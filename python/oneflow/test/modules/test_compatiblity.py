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

log_save = os.getenv("ONEFLOW_TEST_LOG_SAVE") is not None
if log_save:
    # default config
    LOG_PATH = "./model_test_results"
    TAG = "all_test"

    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)

    if os.path.exists(LOG_PATH + "/" + TAG + ".txt"):
        os.remove(LOG_PATH + "/" + TAG + ".txt")

    RESULT_PATH = LOG_PATH + "/" + TAG + ".txt"
else:
    RESULT_PATH = None

@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestApiCompatiblity(flow.unittest.TestCase):
    # alexnet test
    def test_alexnet_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_alexnet.py", "alexnet", "cuda", RESULT_PATH
        )

    # resnet test
    def test_resnet18_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet18", "cuda", RESULT_PATH
        )
    def test_resnet34_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet34", "cuda", RESULT_PATH
        )
    def test_resnet50_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet50", "cuda", RESULT_PATH
        )
    def test_resnet101_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet101", "cuda", RESULT_PATH
        )
    def test_resnet152_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnet152", "cuda", RESULT_PATH
        )
        
    # resnext test
    def test_resnext50_32x4d_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnext50_32x4d", "cuda", RESULT_PATH
        )
    def test_resnext101_32x8d_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "resnext101_32x8d", "cuda", RESULT_PATH
        )

    # wide resnet test
    def test_wide_resnet50_2_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "wide_resnet50_2", "cuda", RESULT_PATH
        )
    def test_wide_resnet101_2_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_resnet.py", "wide_resnet101_2", "cuda", RESULT_PATH
        )

    # mnasnet test
    def test_mnasnet0_5_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet0_5", "cuda", RESULT_PATH
        )
    def test_mnasnet0_75_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet0_75", "cuda", RESULT_PATH
        )
    def test_mnasnet1_0_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet1_0", "cuda", RESULT_PATH
        )
    def test_mnasnet1_3_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mnasnet.py", "mnasnet1_3", "cuda", RESULT_PATH
        )

    # vgg test
    def test_vgg11_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg11", "cuda", RESULT_PATH
        )
    def test_vgg13_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg13", "cuda", RESULT_PATH
        )
    def test_vgg16_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg16", "cuda", RESULT_PATH
        )
    def test_vgg19_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg19", "cuda", RESULT_PATH
        )
    def test_vgg11_bn_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg11_bn", "cuda", RESULT_PATH
        )
    def test_vgg13_bn_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg13_bn", "cuda", RESULT_PATH
        )
    def test_vgg16_bn_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg16_bn", "cuda", RESULT_PATH
        )
    def test_vgg19_bn_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_vgg.py", "vgg19_bn", "cuda", RESULT_PATH
        )
    # squeezenet test
    def test_squeezenet1_0_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_squeezenet.py", "squeezenet1_0", "cuda", RESULT_PATH
        )
    def test_squeezenet1_1_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_squeezenet.py", "squeezenet1_1", "cuda", RESULT_PATH
        )

    # densenet test
    def test_densenet121_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_densenet.py", "densenet121", "cuda", RESULT_PATH
        )
    def test_densenet169_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_densenet.py", "densenet169", "cuda", RESULT_PATH
        )
    def test_densenet201_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_densenet.py", "densenet201", "cuda", RESULT_PATH
        )
    def test_densenet161_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_densenet.py", "densenet161", "cuda", RESULT_PATH
        )

    # mobilenetv2 test
    def test_mobilenetv2_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_mobilenetv2.py", "mobilenet_v2", "cuda", RESULT_PATH
        )

    # shufflenetv2 test
    def test_shufflenet_v2_x0_5_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_shufflenetv2.py", "shufflenet_v2_x0_5", "cuda", RESULT_PATH
        )
    def test_shufflenet_v2_x1_0_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_shufflenetv2.py", "shufflenet_v2_x1_0", "cuda", RESULT_PATH
        )
    def test_shufflenet_v2_x1_5_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_shufflenetv2.py", "shufflenet_v2_x1_5", "cuda", RESULT_PATH
        )
    def test_shufflenet_v2_x2_0_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_shufflenetv2.py", "shufflenet_v2_x2_0", "cuda", RESULT_PATH
        )

    # mobilenetv3 test
    # def test_mobilenet_v3_large_compatiblity(test_case):
    #     test_train_loss_oneflow_pytorch(
    #         test_case, "pytorch_mobilenetv3.py", "mobilenet_v3_large", "cuda"
    #     )
    # def test_mobilenet_v3_small_compatiblity(test_case):
    #     test_train_loss_oneflow_pytorch(
    #         test_case, "pytorch_mobilenetv3.py", "mobilenet_v3_small", "cuda"
    #     )

    # # googlenet test
    # def test_googlenet_compatiblity(test_case):
    #     test_train_loss_oneflow_pytorch(
    #         test_case, "pytorch_googlenet.py", "googlenet", "cuda"
    #     )

    # inceptionv3 test
    # def test_inceptionv3_compatiblity(test_case):
    #     test_train_loss_oneflow_pytorch(
    #         test_case, "pytorch_inceptionv3.py", "inception_v3", "cuda"
    #     )


if __name__ == "__main__":
    unittest.main()
