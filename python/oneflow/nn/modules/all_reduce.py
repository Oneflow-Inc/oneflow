import oneflow as flow
from oneflow.nn.module import Module

from typing import Sequence


class AllReduce(Module):
    def __init__(self, sorted_ranks: Sequence[int]):
        super().__init__()
        self._op = (flow.builtin_op("eager_nccl_all_reduce")
            .Input("in")
            .Output("out")
            .Attr("sorted_ranks", sorted_ranks)
            .Build())

    def forward(self, x):
        assert x.device.type == "cuda"
        assert x.device.index == flow.framework.distribute.get_local_rank()
        return self._op(x)[0]



