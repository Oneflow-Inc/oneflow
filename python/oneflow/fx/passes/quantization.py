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
    conv_weight = []

    def run_node(self, n : Node) -> Any:
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
        return (self.insert_place, self.conv_weight)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if isinstance(submod, flow.nn.Conv2d):
            self.insert_place.append(target)
            self.conv_weight.append(submod.weight)
        return submod(*args, **kwargs)

class QConv2d(flow.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super(QConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.min_max_observer = flow.nn.MinMaxObserver()
        self.fake_quantization = flow.nn.FakeQuantization()

    def forward(self, x):
        scale, zero_point = self.min_max_observer(x)
        x = self.fake_quantization(x, scale, zero_point)
        weight_scale, weight_zero_point = self.min_max_observer(self.weight)
        self.weight = self.fake_quantization(self.weight, weight_scale, weight_zero_point)
        return flow.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

def qat(gm : GraphModule, input) -> GraphModule:
    insert_place, conv_weight = GetInsertNode(gm).propagate(input)
    print(conv_weight)
    
    cnt = 0
    for x in gm.graph.nodes:
        if x.target in insert_place:
            y = x._next
            with gm.graph.inserting_after(x):
                gm.add_submodule("features.conv2d", QConv2d(24, 48, 3))
                # fake_weight : flow.fx.Node = gm.graph.call_module("flow.", args=(x, ), )
                # # fake_weight : flow.fx.Node = gm.graph.call_function(the_function=flow.neg, args=(x, ))
                # _, *nxt_args = y.args
                # y.args = (fake_weight, *nxt_args)
                qconv = gm.graph.call_module(module_name="features.conv2d", args=x.args)
            x.replace_all_uses_with(qconv)
            gm.graph.erase_node(x)

    gm.recompile()
    return gm
