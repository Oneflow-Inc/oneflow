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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op


class Arange(Module):
    def __init__(self, start, end, step=1) -> None:
        super().__init__()
        self.start = 0 if start is None else start
        self.end = 1 if end is None else end
        self.step = step
        self.dtype = flow.int64  # "Only support dtype: `flow.int64` for now!"
        assert self.end > self.start, "end should be larger than start"
        assert self.step <= self.end - self.start, "step is ilegal"
        assert type(self.start) == int, "Params `start`'s type should be int"
        assert type(self.end) == int, "Params `end`'s type should be int"
        assert type(self.step) == int, "Params `step`'s type should be int"

        self._op_arange = flow.builtin_op("range").Output("out").Build()

    def forward(self):
        return self._op_arange(
            start=self.start, delta=self.step, limit=self.end, dtype=self.dtype
        )[0]


@oneflow_export("arange")
def arange_op(start=1, end=1, step=1):
    r"""
    Returns a 1-D tensor of size :math:`\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1`
    with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
    the gap between two values in the tensor.

    .. math::
    \text{out}_{i+1} = \text{out}_i + \text{step}.

    Args:
    start (float): the starting value for the set of points. Default: ``0``.
    end (float): the ending value for the set of points
    step (float): the gap between each pair of adjacent points. Default: ``1``.

    Keyword args:
    dtype: If `dtype` is not given, the `dtype` is inferred to be the default dtype.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        y = flow.arange(0, 5)
        # [0, 1, 2, 3, 4]

    """
    return Arange(start, end, step)()
