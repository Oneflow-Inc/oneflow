from typing import Union

import numpy as np
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import Tensor, register_tensor_op


class Split(Module):
    def __init__(self, size_or_sections: Union[int, list], dim: int = 0) -> None:
        super().__init__()
        assert isinstance(size_or_sections, (int, list))
        if isinstance(size_or_sections, list):
            self.split_count = 0
            for item in size_or_sections:
                assert isinstance(item, int)
                self.split_count += item
            self._op = flow.builtin_op("split_sizes").Input("in").Output("out").Attr("axis", dim).Attr("sizes",
                                                                                                       size_or_sections).Build()
        else:
            self._op = flow.builtin_op("split").Input("in").Output("out").Attr("axis", dim).Attr("sections",
                                                                                                 size_or_sections).Build()
        self.dim = dim

    def forward(self, input):
        assert isinstance(input, Tensor)
        input_shape = input.shape
        max_dim = len(input_shape)

        # The axis must be in range [-(_max_dim +1), _max_dim]
        if self.dim < 0:
            self.dim = self.dim + max_dim + 1
        assert (self.dim >= 0) and (self.dim <= max_dim)
        if hasattr(self, "split_count"):
            assert self.split_count == input_shape[self.dim]

        return self._op(input)


@oneflow_export("split")
@register_tensor_op("split")
@experimental_api
def split(input: Tensor, size_or_sections: Union[int, list], dim: int = 0) -> None:
    return Split(size_or_sections, dim)(input)
