from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from typing import Optional

class Expand(Module):

    def __init__(self, *sizes) -> None:
        super().__init__()
        self.expand_size = list(*sizes)

    def forward(self, x):
        if x.dtype == flow.int8:
            x = flow.experimental.cast(x, flow.int32)
        expand_size = self.expand_size
        assert len(expand_size) >= len(x.shape), 'The desired expanded dims should not be less than the input dims.'
        original_stride = [1]
        for i in range(len(x.shape) - 2, -1, -1):
            original_stride.insert(0, original_stride[0] * x.shape[i + 1])
        new_size = []
        new_stride = []
        diff = len(expand_size) - len(x.shape)
        for i in range(len(expand_size) - 1, -1, -1):
            if i >= diff:
                if expand_size[i] == -1 or expand_size[i] == x.shape[i - diff]:
                    new_size.insert(0, x.shape[i - diff])
                    new_stride.insert(0, original_stride[i - diff])
                else:
                    assert expand_size[i] >= 1 and x.shape[i - diff] == 1
                    new_size.insert(0, expand_size[i])
                    new_stride.insert(0, 0)
            else:
                assert expand_size[i] >= 1
                new_size.insert(0, expand_size[i])
                if expand_size[i] == 1:
                    new_stride.insert(0, new_stride[0])
                else:
                    new_stride.insert(0, 0)
        return flow.F.expand(x, in_shape=list(x.shape), out_shape=new_size, stride=new_stride)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)