import oneflow as flow
from typing import Dict
import oneflow.fx
from oneflow.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional
from oneflow.fx._compatibility import compatibility
from ..node import Argument, Target


@compatibility(is_backward_compatible=True)
class QuantizationAwareTraining(flow.fx.Interpreter):
    
    insert_place = []

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
        return self.insert_place

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if isinstance(submod, flow.nn.Conv2d):
            self.insert_place.append(target)
        return submod(*args, **kwargs)

