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
import collections.abc
import tempfile
import unittest
from itertools import repeat
from typing import Tuple, Union, List
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


def np_relu(np_arr):
    return np.where(np_arr > 0, np_arr, 0)


class TestModule(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_nested_module(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = flow.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        m = CustomModule()
        x = flow.Tensor(2, 3)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        y = m(x)
        test_case.assertTrue(np.array_equal(np_relu(x.numpy()), y.numpy()))

    @flow.unittest.skip_unless_1n1d()
    def test_relu(test_case):
        relu = flow.nn.ReLU()
        x = flow.Tensor(2, 3)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        y = relu(x)
        test_case.assertTrue(np.array_equal(np_relu(x.numpy()), y.numpy()))

    @flow.unittest.skip_unless_1n1d()
    def test_load_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = flow.nn.Parameter(flow.Tensor(2, 3))

            def forward(self, x):
                return self.w

        m = CustomModule()
        ones = np.ones((2, 3), dtype=np.float32)
        m.load_state_dict({"w": ones})
        x = flow.Tensor(2, 3)
        y = m(x).numpy()
        test_case.assertTrue(np.array_equal(y, ones))

    @flow.unittest.skip_unless_1n1d()
    def test_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        tensor0 = flow.nn.Parameter(flow.Tensor(2, 3))
        tensor1 = flow.nn.Parameter(flow.Tensor(2, 3))
        sub_module = CustomModule(tensor0, tensor1)
        m = CustomModule(tensor1, sub_module)
        state_dict = m.state_dict()
        test_case.assertEqual(
            state_dict,
            {"param2.param1": tensor0, "param2.param2": tensor1, "param1": tensor1},
        )

    @flow.unittest.skip_unless_1n1d()
    def test_parameter(test_case):
        shape = (3, 4)
        t = flow.Tensor(*shape)
        p = flow.nn.Parameter(t)
        test_case.assertEqual(type(p), flow.nn.Parameter)
        test_case.assertEqual(p.shape, shape)

    @flow.unittest.skip_unless_1n1d()
    def test_module_forward(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, x):
                return x + self.w

        m = CustomModule(5)
        test_case.assertEqual(m(1), 6)
        m = CustomModule(4)
        test_case.assertEqual(m(3), 7)

    @flow.unittest.skip_unless_1n1d()
    def test_train_eval(test_case):
        m = flow.nn.Module()
        test_case.assertEqual(m.training, True)
        m.train()
        test_case.assertEqual(m.training, True)
        m.eval()
        test_case.assertEqual(m.training, False)

    @flow.unittest.skip_unless_1n1d()
    def test_module_setattr(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        param0 = flow.nn.Parameter(flow.Tensor(2, 3))
        param1 = flow.nn.Parameter(flow.Tensor(2, 3))
        param2 = CustomModule(param0, param1)
        m = CustomModule(param1, param2)
        params = list(m.parameters())
        test_case.assertEqual(len(params), 2)

        test_case.assertTrue(
            np.allclose(params[0].numpy(), param1.numpy(), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(params[1].numpy(), param0.numpy(), atol=1e-4, rtol=1e-4)
        )
        children = list(m.children())
        test_case.assertEqual(len(children), 1)
        child = children[0]
        test_case.assertEqual(child, param2)
        child_params = list(child.parameters())

        test_case.assertEqual(len(child_params), 2)
        test_case.assertTrue(np.allclose(child_params[0].numpy(), param0.numpy()))
        test_case.assertTrue(np.allclose(child_params[1].numpy(), param1.numpy()))

    @flow.unittest.skip_unless_1n1d()
    def test_module_apply(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.modules = flow.nn.Module()

        global module_num
        module_num = 0

        def get_module_num(m):
            global module_num
            module_num += 1

        net = CustomModule()
        net.apply(get_module_num)
        test_case.assertEqual(module_num, 2)

    @flow.unittest.skip_unless_1n1d()
    def test_save_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = flow.nn.Parameter(flow.Tensor(32, 1024, 1024))
                self.param2 = flow.nn.Parameter(flow.Tensor(32, 1024, 1024))

            def forward(self):
                return self.param1 + self.param2

        m = CustomModule()
        res1 = m()
        state_dict = m.state_dict()
        with tempfile.TemporaryDirectory() as save_dir:
            flow.save(state_dict, save_dir)
            loaded_state_dict = flow.load(save_dir)
            m.load_state_dict(loaded_state_dict)
        res2 = m()
        test_case.assertTrue(np.array_equal(res1.numpy(), res2.numpy()))

    @flow.unittest.skip_unless_1n2d()
    def test_save_and_load_consistent(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = flow.nn.Parameter(flow.randn(3, 32, 3, 3))

            def forward(self):
                return self.param

        m = CustomModule()
        m = m.to_consistent(flow.placement("cuda", {0: range(2)}), flow.sbp.broadcast)
        res1 = m()
        state_dict = m.state_dict()

        with tempfile.TemporaryDirectory() as f:
            with test_case.assertRaises(Exception):
                flow.save(state_dict, f)

            consistent_src_dst_rank = 0
            flow.save(state_dict, f, consistent_dst_rank=consistent_src_dst_rank)
            rank = flow.framework.distribute.get_rank()
            if rank != consistent_src_dst_rank:
                test_case.assertEqual(len(os.listdir(f)), 0)

            m = CustomModule()
            m = m.to_consistent(
                flow.placement("cuda", {0: range(2)}), flow.sbp.broadcast
            )

            with test_case.assertRaises(Exception):
                loaded_state_dict = flow.load(f)
                m.load_state_dict(loaded_state_dict)

            loaded_state_dict = flow.load(
                f, consistent_src_rank=consistent_src_dst_rank
            )
            test_case.assertEqual(len(loaded_state_dict), 1)
            test_case.assertEqual(list(loaded_state_dict.keys())[0], "param")
            m.load_state_dict(loaded_state_dict)
            res2 = m()

        test_case.assertTrue(
            np.array_equal(
                res1.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy(),
                res2.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy(),
            )
        )
    
    @flow.unittest.skip_unless_1n1d()
    def test_moduledict(test_case):
        class _DenseLayer(nn.Module):
            def __init__(
                self,
                num_input_features: int,
                growth_rate: int,
                bn_size: int,
                drop_rate: float,
            ) -> None:
                super(_DenseLayer, self).__init__()
                self.norm1: nn.BatchNorm2d
                self.add_module('norm1', nn.BatchNorm2d(num_input_features))
                self.relu1: nn.ReLU
                self.add_module('relu1', nn.ReLU(inplace=True))
                self.conv1: nn.Conv2d
                self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                                growth_rate, kernel_size=1, stride=1,
                                                bias=False))
                self.norm2: nn.BatchNorm2d
                self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
                self.relu2: nn.ReLU
                self.add_module('relu2', nn.ReLU(inplace=True))
                self.conv2: nn.Conv2d
                self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                                kernel_size=3, stride=1, padding=1,
                                                bias=False))
                self.drop_rate = float(drop_rate)
            
            def bn_function(self, inputs: List[flow.Tensor]) -> flow.Tensor:
                concated_features = flow.cat(inputs, 1)
                bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
                return bottleneck_output

            # todo: rewrite when torchscript supports any
            def any_requires_grad(self, input: List[flow.Tensor]) -> bool:
                    for tensor in input:
                        if tensor.requires_grad:
                            return True
                    return False
        
            def forward(self, input: List[flow.Tensor]) -> flow.Tensor:
                pass

            def forward(self, input: flow.Tensor) -> flow.Tensor:
                pass

            def forward(self, input) -> flow.Tensor:  # noqa: F811
                if isinstance(input, flow.Tensor):
                    prev_features = [input]
                else:
                    prev_features = input

                bottleneck_output = self.bn_function(prev_features)

                new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
                if self.drop_rate > 0:
                    new_features = flow.F.dropout(new_features, p=self.drop_rate,
                                            training=self.training)
                return new_features

        class _DenseBlock(nn.ModuleDict):
            _version = 2

            def __init__(
                self,
                num_layers: int,
                num_input_features: int,
                bn_size: int,
                growth_rate: int,
                drop_rate: float,
            ) -> None:
                super(_DenseBlock, self).__init__()
                for i in range(num_layers):
                    layer = _DenseLayer(
                        num_input_features + i * growth_rate,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        drop_rate=drop_rate,
                    )
                    self.add_module('denselayer%d' % (i + 1), layer)

            def forward(self, init_features):
                features = [init_features]
                for name, layer in self.items():
                    new_features = layer(features)
                    features.append(new_features)
                return flow.cat(features, dim=1)
        
        class _Transition(nn.Sequential):
            def __init__(self, num_input_features: int, num_output_features: int) -> None:
                super(_Transition, self).__init__()
                self.add_module('norm', nn.BatchNorm2d(num_input_features))
                self.add_module('relu', nn.ReLU(inplace=True))
                self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                                kernel_size=1, stride=1, bias=False))
                self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

        class DenseNet(flow.nn.Module):
            def __init__(
                self,
                growth_rate: int = 32,
                block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                num_init_features: int = 64,
                bn_size: int = 4,
                drop_rate: float = 0,
                num_classes: int = 1000,
            ):
                super().__init__()
                # First convolution
                self.features = nn.Sequential(OrderedDict([
                    ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                        padding=3, bias=False)),
                    ('norm0', nn.BatchNorm2d(num_init_features)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]))

                # Each denseblock
                num_features = num_init_features
                for i, num_layers in enumerate(block_config):
                    block = _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                    )
                    self.features.add_module('denseblock%d' % (i + 1), block)
                    num_features = num_features + num_layers * growth_rate
                    if i != len(block_config) - 1:
                        trans = _Transition(num_input_features=num_features,
                                            num_output_features=num_features // 2)
                        self.features.add_module('transition%d' % (i + 1), trans)
                        num_features = num_features // 2

                    def forward(self):
                        return self.param
                
                # Final batch norm
                self.features.add_module('norm5', nn.BatchNorm2d(num_features))

                # Linear layer
                self.classifier = nn.Linear(num_features, num_classes)

                # Official init from torch repo.
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.constant_(m.bias, 0)

            def forward(self, x: flow.Tensor) -> flow.Tensor:
                features = self.features(x)
                out = flow.F.relu(features, inplace=True)
                out = flow.F.adaptive_avg_pool2d(out, (1, 1))
                out = flow.flatten(out, 1)
                out = self.classifier(out)
                return out

        model = DenseNet(32, (6, 12, 24, 16), 64)
        input = flow.tensor(np.random.randn(1, 3, 224, 224), dtype=flow.float32)
        output = model(input)
        test_case.assertEqual(output.shape, flow.Size([1, 1000]))


if __name__ == "__main__":
    unittest.main()
