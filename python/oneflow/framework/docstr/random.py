import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.F.bernoulli,
    "\n    bernoulli(input, *, generator=None, out=None)\n    \n    This operator returns a Tensor with binaray random numbers (0 / 1) from a Bernoulli distribution.\n\n    Args:\n        input (Tensor): the input tensor of probability values for the Bernoulli distribution\n        generator: (Generator, optional) a pseudorandom number generator for sampling\n        out (Tensor, optional): the output tensor.\n\n    Shape:\n        - Input: :math:`(*)`. Input can be of any shape\n        - Output: :math:`(*)`. Output is of the same shape as input\n\n    For example:\n\n    .. code-block:: python\n\n        >>> import numpy as np\n        >>> import oneflow as flow\n        >>> arr = np.array(\n        ...    [\n        ...        [1.0, 1.0, 1.0],\n        ...        [1.0, 1.0, 1.0],\n        ...        [1.0, 1.0, 1.0],\n        ...    ]\n        ... )\n        >>> x = flow.Tensor(arr)\n        >>> y = flow.F.bernoulli(x)\n        >>> y\n        tensor([[1., 1., 1.],\n                [1., 1., 1.],\n                [1., 1., 1.]], dtype=oneflow.float32)\n\n    ",
)
