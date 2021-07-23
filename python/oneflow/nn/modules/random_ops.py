import sys
import random
import oneflow as flow
from oneflow.nn.module import Module

def bernoulli(input, *, generator=None, out=None):
    """This operator returns a Tensor with binaray random numbers (0 / 1) from a Bernoulli distribution.

    Args:
        input(Tensor) - the input tensor of probability values for the Bernoulli distribution
        generator: (optional) – a pseudorandom number generator for sampling
        out (Tensor, optional) – the output tensor.

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> arr = np.array(
        ...    [
        ...        [1.0, 1.0, 1.0],
        ...        [1.0, 1.0, 1.0],
        ...        [1.0, 1.0, 1.0],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = flow.bernoulli(x)
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)


    """
    return flow.F.bernoulli(input, flow.float32, generator)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)