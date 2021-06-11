import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import Tensor


class Randperm(Module):
    def __init__(self, generator=None, dtype=flow.int64, layout=None, device=None, requires_grad=False,
                 pin_memory=False) -> None:
        super().__init__()
        if generator is not None:
            print(
                "WARNING:",
                "oneflow.randperm.generator",
                "will not be used. Custom generator is not currently supported."
            ),
        if layout is not None:
            print(
                "WARNING:",
                "oneflow.randperm.layout",
                "will not be used. Layout is not currently supported."
            ),
        if isinstance(device, str):
            device = flow.device(device)
        else:
            device = device if device is not None else flow.device("cpu")

        assert isinstance(device, flow.device)
        assert isinstance(dtype, flow.dtype)
        assert isinstance(requires_grad, bool)
        assert isinstance(pin_memory, bool)

        # self.device = device
        # self.dtype = dtype
        # self.requires_grad = requires_grad
        # self.pin_memory = pin_memory
        self._op = (
            flow.builtin_op("randperm")
                .Input("n")
                .Output("out")
                .Attr("device", device)
                .Attr("dtype", dtype)
                .Attr("requires_grad", requires_grad)
                .Attr("pin_memory", pin_memory)
                .Build()
        )

    def forward(self, n, out=None):
        res = self._op(n)[0]
        if out is not None:
            assert isinstance(out, Tensor)
            out = res
        return res


@oneflow_export("randperm")
@register_tensor_op("randperm")
@experimental_api
def randperm(n, generator=None, out=None, dtype=flow.int64, layout=None, device=None, requires_grad=False,
             pin_memory=False) -> Tensor:
    r"""
    Returns a random permutation of integers from ``0`` to ``n - 1``.

    Args:
        n (int): the upper bound (exclusive)

    Keyword args:
        {generator}: custom generator is not currently supported.
        out (Tensor): output Tensor.
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: ``oneflow.int64``.
        {layout}: layout is not currently supported.
        {device}
        {requires_grad}
        {pin_memory}

    Example::

    .. code-block:: python

        >>> torch.randperm(4)
        tensor([2, 1, 0, 3])
    """
    return Randperm(generator, dtype, layout, device, requires_grad, pin_memory)(n, out)
