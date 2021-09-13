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
import oneflow as flow
from typing import Dict
import oneflow.fx
from oneflow.fx.graph_module import GraphModule
from oneflow.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional
from oneflow.fx._compatibility import compatibility
from ..node import Argument, Target


@compatibility(is_backward_compatible=True)
class GetInsertNode(flow.fx.Interpreter):

    insert_place = []
    conv_state = []

    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        return getattr(self, n.op)(n.target, args, kwargs)

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        self.insert_place.clear()
        super().run(*args)
        return (self.insert_place, self.conv_state)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if isinstance(submod, flow.nn.Conv2d):
            self.insert_place.append(target)
            self.conv_state.append(submod)
        return submod(*args, **kwargs)


class QConv2d(flow.nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups
    ) -> None:
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        )
        self.min_max_observer = flow.nn.MinMaxObserver()
        self.fake_quantization = flow.nn.FakeQuantization()

    def forward(self, x):
        scale, zero_point = self.min_max_observer(x)
        x = self.fake_quantization(x, scale, zero_point)
        weight_scale, weight_zero_point = self.min_max_observer(self.weight)
        self.weight = flow.nn.Parameter(self.fake_quantization(
            self.weight, weight_scale, weight_zero_point
        ))
        return flow.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

def get_current_module_space(mod : str):
    x = mod.split(".")
    x_len = len(x)
    y = ""
    for _ in range(x_len - 1):
        y += x[_]
        if _ < x_len - 2:
            y += "."
    return y

def qat(gm: GraphModule, input) -> GraphModule:
    insert_place, conv_state = GetInsertNode(gm).propagate(input)
    cnt = 0
    for x in gm.graph.nodes:
        if x.target in insert_place:
            with gm.graph.inserting_after(x):
                gm.add_submodule(
                    f"{get_current_module_space(x.target)}.fake_conv2d.{cnt}",
                    QConv2d(
                        conv_state[cnt].in_channels,
                        conv_state[cnt].out_channels,
                        conv_state[cnt].kernel_size,
                        conv_state[cnt].stride,
                        conv_state[cnt].padding,
                        conv_state[cnt].dilation,
                        conv_state[cnt].groups,
                    ),
                )
                qconv = gm.graph.call_module(
                    module_name=f"{get_current_module_space(x.target)}.fake_conv2d.{cnt}", args=x.args
                )
                cnt = cnt + 1
            x.replace_all_uses_with(qconv)
            gm.graph.erase_node(x)

    gm.recompile()
    return gm
