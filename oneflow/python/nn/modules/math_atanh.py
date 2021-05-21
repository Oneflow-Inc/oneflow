class Ceil(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("ceil").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]

    
    
@oneflow_export("ceil")
@register_tensor_op("ceil")
@experimental_api
def ceil_op(x):
    r"""ceil(input, out=None) -> Tensor

    Returns a new tensor with the ceil of the elements of :attr:`input`,
    the smallest integer greater than or equal to each element.

    .. math::
        \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1

    Args:
        input (Tensor): the input tensor.
        out (Tensor, optional): the output tensor.

    Example::

        >>> a = torch.randn(4)
        >>> a
        tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        >>> torch.ceil(a)
        tensor([-0., -1., -1.,  1.])
    Type:      builtin_function_or_method
            
    """
    return Ceil()(x)