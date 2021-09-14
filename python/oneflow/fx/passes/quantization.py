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
from oneflow.fx.passes.quantization_ops.linear import QLinear
from ..node import Argument, Target
from .quantization_ops.conv import QConv2d

trace_op = (
    flow.nn.Conv2d,
    flow.nn.Linear,
)


@compatibility(is_backward_compatible=True)
class GetInsertNode(flow.fx.Interpreter):

    insert_place = []
    insert_op_state = []

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
        return (self.insert_place, self.insert_op_state)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if isinstance(submod, trace_op):
            self.insert_place.append(target)
            self.insert_op_state.append(submod)
        return submod(*args, **kwargs)


def get_current_module_space(mod: str):
    x = mod.split(".")
    x_len = len(x)
    y = ""
    for _ in range(x_len - 1):
        y += x[_]
        if _ < x_len - 2:
            y += "."
    return y


fx_qat_name = dict()


def quantization_aware_training(gm: GraphModule, input, qconfig: dict) -> GraphModule:
    fx_qat_name.clear()

    quantization_bit = 8
    quantization_scheme = "symmetric"
    quantization_formula = "google"
    per_layer_quantization = True
    if "quantization_bit" in qconfig:
        quantization_bit = qconfig["quantization_bit"]
    if "quantization_scheme" in qconfig:
        quantization_scheme = qconfig["quantization_scheme"]
    if "quantization_formula" in qconfig:
        quantization_formula = qconfig["quantization_formula"]
    if "per_layer_quantization" in qconfig:
        per_layer_quantization = qconfig["per_layer_quantization"]

    insert_place, insert_op_state = GetInsertNode(gm).propagate(input)
    cnt = 0
    for x in gm.graph.nodes:
        if x.target in insert_place:
            with gm.graph.inserting_after(x):
                if isinstance(insert_op_state[cnt], flow.nn.Conv2d):
                    gm.add_submodule(
                        f"{get_current_module_space(x.target)}.fake_conv2d.{cnt}",
                        QConv2d(
                            insert_op_state[cnt].in_channels,
                            insert_op_state[cnt].out_channels,
                            insert_op_state[cnt].kernel_size,
                            insert_op_state[cnt].stride,
                            insert_op_state[cnt].padding,
                            insert_op_state[cnt].dilation,
                            insert_op_state[cnt].groups,
                            quantization_bit,
                            quantization_scheme,
                            quantization_formula,
                            per_layer_quantization,
                        ),
                    )
                    qconv = gm.graph.call_module(
                        module_name=f"{get_current_module_space(x.target)}.fake_conv2d.{cnt}",
                        args=x.args,
                    )
                    fx_qat_name[
                        str(x.target)
                    ] = f"{get_current_module_space(x.target)}.fake_conv2d.{cnt}"
                    cnt = cnt + 1
                    x.replace_all_uses_with(qconv)
                    gm.graph.erase_node(x)
                elif isinstance(insert_op_state[cnt], flow.nn.Linear):
                    bias = True
                    if insert_op_state[cnt].bias is None:
                        bias = False
                    gm.add_submodule(
                        f"{get_current_module_space(x.target)}.fake_matmul.{cnt}",
                        QLinear(
                            insert_op_state[cnt].in_features,
                            insert_op_state[cnt].out_features,
                            bias,
                            quantization_bit,
                            quantization_scheme,
                            quantization_formula,
                            per_layer_quantization,
                        ),
                    )
                    qmatmul = gm.graph.call_module(
                        module_name=f"{get_current_module_space(x.target)}.fake_matmul.{cnt}",
                        args=x.args,
                    )
                    fx_qat_name[
                        str(x.target)
                    ] = f"{get_current_module_space(x.target)}.fake_matmul.{cnt}"
                    cnt = cnt + 1
                    x.replace_all_uses_with(qmatmul)
                    gm.graph.erase_node(x)

    gm.recompile()
    return gm


def convert_quantization_aware_training_module(
    gm: GraphModule, origin: oneflow.nn.Module, input
):
    assert (
        len(fx_qat_name) > 0
    ), "quantization_aware_training must be called before convert_quantization_aware_training_module"
    insert_place, insert_op_state = GetInsertNode(origin).propagate(input)

    origin_w = origin.state_dict()
    qat_w = gm.state_dict()
    for k, v in qat_w.items():
        if k in insert_place:
            origin_w[fx_qat_name[k]] = v
    return
