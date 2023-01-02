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

import numpy as np
import oneflow as flow
import oneflow.unittest


@unittest.skip(reason="network fluctuations can cause downloads to fail!")
class TestHub(flow.unittest.TestCase):
    def test_hub_list_api(test_case):
        entrypoints = flow.hub.list("OneFlow-Inc/vision", force_reload=False)
        test_case.assertEqual("alexnet" in entrypoints, True)
        test_case.assertEqual("densenet121" in entrypoints, True)

    def test_hub_help_api(test_case):
        help_info = flow.hub.help("Oneflow-Inc/vision", "resnet18", force_reload=False)
        print(help_info)

    def test_hub_load_api(test_case):
        repo = "Oneflow-Inc/vision"
        model = flow.hub.load(repo, "resnet18", pretrained=True)
        x = flow.randn(1, 3, 224, 224)
        y = model(x)
        test_case.assertTrue(np.array_equal(y.size(), (1, 1000)))

    def test_hub_download_url_to_file__api(test_case):
        flow.hub.download_url_to_file(
            "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet18.zip",
            "/tmp/temporary_file",
        )

    def test_hub_load_state_dict_from_url_api(test_case):
        state_dict = flow.hub.load_state_dict_from_url(
            "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet18.zip"
        )
        test_case.assertEqual("layer3.1.bn2.bias" in state_dict.keys(), True)


if __name__ == "__main__":
    unittest.main()
