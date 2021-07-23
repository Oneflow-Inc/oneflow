from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.ops.transpose_util import get_perm_when_transpose_axis_to_last_dim, get_inversed_perm

class Sort(Module):

    def __init__(self, dim: int=-1, descending: bool=False) -> None:
        super().__init__()
        self.dim = dim
        direction = 'DESCENDING' if descending else 'ASCENDING'
        self._argsort_op = flow.builtin_op('arg_sort').Input('in').Output('out').Attr('direction', direction).Build()

    def forward(self, input):
        num_dims = len(input.shape)
        dim = self.dim if self.dim >= 0 else self.dim + num_dims
        assert 0 <= dim < num_dims, 'dim out of range'
        if dim == num_dims - 1:
            indices = self._argsort_op(input)[0]
            return (flow.experimental.gather(input, indices, dim), indices)
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_dims, dim)
            x = flow.F.transpose(input, perm=perm)
            indices = self._argsort_op(x)[0]
            indices = flow.F.transpose(indices, perm=get_inversed_perm(perm))
            return (flow.experimental.gather(input, indices, dim), indices)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)