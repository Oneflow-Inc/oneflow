from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from typing import Optional, Sequence

class Permute(Module):

    def __init__(self, *dims) -> None:
        super().__init__()
        self.perm = list(*dims)

    def forward(self, x):
        assert len(self.perm) == len(x.shape)
        new_perm = []
        for dim in self.perm:
            if dim < 0:
                dim += len(self.perm)
            assert dim >= 0 and dim < len(x.shape), 'Invalid dim0 {}, len(shape): {}'.format(dim, len(x.shape))
            new_perm.append(dim)
        return flow.F.transpose(x, perm=new_perm)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)