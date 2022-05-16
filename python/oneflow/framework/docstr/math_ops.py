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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.abs,
    r"""Return the absolute value of each element in input tensor:math:`y = |x|` element-wise.

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.tensor(np.array([-1, 2, -3, 4]).astype(np.float32))
        >>> flow.abs(x)
        tensor([1., 2., 3., 4.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.add,
    r"""Computes the addition of `input` by `other` for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = input + other

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        # element-wise add
        >>> x = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # scalar add
        >>> x = 5
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # broadcast add
        >>> x = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

    """,
)

add_docstr(
    oneflow.floor,
    """floor(input)
    Returns a new tensor with the arcsine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\lfloor \\text{input}_{i} \\rfloor

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([-0.5,  1.5, 0,  0.8]), dtype=flow.float32)
        >>> output = flow.floor(input)
        >>> output.shape
        oneflow.Size([4])
        >>> output.numpy()
        array([-1.,  1.,  0.,  0.], dtype=float32)

        >>> input1 = flow.tensor(np.array([[0.8, 1.0], [-0.6, 2.5]]), dtype=flow.float32)
        >>> output1 = input1.floor()
        >>> output1.shape
        oneflow.Size([2, 2])
        >>> output1.numpy()
        array([[ 0.,  1.],
               [-1.,  2.]], dtype=float32)

    """,
)

add_docstr(
    oneflow.floor_,
    r"""floor_(input)
    In-place version of :func:`oneflow.floor`
    """,
)

add_docstr(
    oneflow.div,
    r"""div(input, other)
    Computes the division of input by other for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = \frac{input}{other}

    Args:
        input (Union[int, float, oneflow.Tensor]): input.
        other (Union[int, float, oneflow.Tensor]): other.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        # element-wise divide
        >>> input = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar divide
        >>> input = 5
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast divide
        >>> input = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

    """,
)

add_docstr(
    oneflow.mul,
    r"""mul(input, other)
    Computes the multiplication of input by other for each element, scalar and broadcast promotation are supported.

    The formula is:

    .. math::
        \text{out}_i = \text{input}_i \times \text{other}_i

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        # element-wise multiply
        >>> input = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar mutiply
        >>> input = 5
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast mutiply
        >>> input = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape
        (2, 3)

    """,
)

add_docstr(
    oneflow.reciprocal,
    r"""reciprocal(x)
    Computes the safe reciprocal of x. If x is zero, the reciprocal will
    be also set to zero.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=flow.float32)
        >>> out = flow.reciprocal(x)
        >>> out.numpy()
        array([[1.        , 0.5       , 0.33333334],
               [0.25      , 0.2       , 0.16666667]], dtype=float32)
    """,
)

add_docstr(
    oneflow.sub,
    r"""sub(input, other)
    Computes the subtraction of input by other for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = input - other

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        # element-wise subtract
        >>> input = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar subtract
        >>> input = 5
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast subtract
        >>> input = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

    """,
)

add_docstr(
    oneflow.asin,
    r"""
    Returns a new tensor with the arcsine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \sin^{-1}(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([-0.5,  0.8, 1.0,  -0.8]), dtype=flow.float32)
        >>> output = flow.asin(input)
        >>> output.shape
        oneflow.Size([4])
        >>> output
        tensor([-0.5236,  0.9273,  1.5708, -0.9273], dtype=oneflow.float32)
        >>> input1 = flow.tensor(np.array([[0.8, 1.0], [-0.6, -1.0]]), dtype=flow.float32)
        >>> output1 = input1.asin()
        >>> output1.shape
        oneflow.Size([2, 2])
        >>> output1
        tensor([[ 0.9273,  1.5708],
                [-0.6435, -1.5708]], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.asinh,
    r"""
    Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \sinh^{-1}(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([2, 3, 4]), dtype=flow.float32)
        >>> output = flow.asinh(input)
        >>> output.shape
        oneflow.Size([3])
        >>> output
        tensor([1.4436, 1.8184, 2.0947], dtype=oneflow.float32)

        >>> input1 = flow.tensor(np.array([[-1, 0, -0.4], [5, 7, 0.8]]), dtype=flow.float32)
        >>> output1 = input1.asinh()
        >>> output1.shape
        oneflow.Size([2, 3])
        >>> output1
        tensor([[-0.8814,  0.0000, -0.3900],
                [ 2.3124,  2.6441,  0.7327]], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.atan,
    r"""atan(input)
    Returns a new tensor with the arctangent of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \tan^{-1}(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([0.5, 0.6, 0.7]), dtype=flow.float32)
        >>> output = flow.atan(input)
        >>> output.shape
        oneflow.Size([3])

    """,
)

add_docstr(
    oneflow.ceil,
    r"""Returns a new tensor with the ceil of the elements of :attr:`input`,
    the smallest integer greater than or equal to each element.

    The equation is:

    .. math::
        \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1

    Args:
        input (oneflow.Tensor): A Tensor.

    Returns:
        oneflow.Tensor: The result Tensor

    For example:


    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.tensor(np.array([0.1, -2, 3.4]).astype(np.float32))
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([3])
        >>> y
        tensor([ 1., -2.,  4.], dtype=oneflow.float32)
        >>> x = flow.tensor(np.array([[2.5, 4.6, 0.6],[7.8, 8.3, 9.2]]).astype(np.float32))
        >>> y = x.ceil()
        >>> y.shape
        oneflow.Size([2, 3])
        >>> y
        tensor([[ 3.,  5.,  1.],
                [ 8.,  9., 10.]], dtype=oneflow.float32)
        >>> x = flow.tensor(np.array([[[2.2, 4.4, 6.5],[7.1, 8.2, 9.3]],[[10.6,11.2,12.2],[13.5,14.8,15.9]]]).astype(np.float32))
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([2, 2, 3])
        >>> y
        tensor([[[ 3.,  5.,  7.],
                 [ 8.,  9., 10.]],
        <BLANKLINE>
                [[11., 12., 13.],
                 [14., 15., 16.]]], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.negative,
    r"""negative(input)
    This operator computes the negative value of Tensor.

    Args:
        input (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.tensor(
        ...    np.array([1.0, -1.0, 2.3]).astype(np.float32), dtype=flow.float32
        ... )
        >>> out = flow.negative(input)
        >>> out
        tensor([-1.0000,  1.0000, -2.3000], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.log1p,
    r"""log1p(input)
    Returns a new tensor with the natural logarithm of (1 + input).

    .. math::
        \text{out}_{i}=\log_e(1+\text{input}_{i})

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.tensor(np.array([1.3, 1.5, 2.7]), dtype=flow.float32)
        >>> out = flow.log1p(x)
        >>> out
        tensor([0.8329, 0.9163, 1.3083], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.neg,
    r"""neg(input)
    Alias for :func:`oneflow.negative`.
    """,
)

add_docstr(
    oneflow.exp,
    r"""exp(x)

    This operator computes the exponential of Tensor.

    The equation is:

    .. math::

        out = e^x

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> y = flow.exp(x)
        >>> y
        tensor([ 2.7183,  7.3891, 20.0855], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.acos,
    r"""
    Returns a new tensor with the inverse cosine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \arccos(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> arr = np.array([0.5, 0.6, 0.7])
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> output = flow.acos(input)
        >>> output
        tensor([1.0472, 0.9273, 0.7954], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.acosh,
    r"""
    Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.

    .. math::

        \text{out}_{i} = \cosh^{-1}(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.tensor(np.array([2, 3, 4]).astype(np.float32))
        >>> out1 = flow.acosh(x1)
        >>> out1
        tensor([1.3170, 1.7627, 2.0634], dtype=oneflow.float32)
        >>> x2 = flow.tensor(np.array([1.5, 2.6, 3.7]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.acosh(x2)
        >>> out2
        tensor([0.9624, 1.6094, 1.9827], device='cuda:0', dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.atanh,
    r"""atanh(input)
    Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \tanh^{-1}(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.tensor(np_arr, dtype=flow.float32)
        >>> output = flow.atanh(input)
        >>> output
        tensor([0.5493, 0.6931, 0.8673], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.sign,
    r"""sign(input)
    Computes the sign of Tensor.

    .. math::

        \text{out}_{i}  = \text{sgn}(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.tensor(np.array([-2, 0, 2]).astype(np.float32))
        >>> out1 = flow.sign(x1)
        >>> out1.numpy()
        array([-1.,  0.,  1.], dtype=float32)
        >>> x2 = flow.tensor(np.array([-3.2, -4.5, 5.8]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.sign(x2)
        >>> out2.numpy()
        array([-1., -1.,  1.], dtype=float32)

    """,
)

add_docstr(
    oneflow.sin,
    r"""sin(x)
    Returns a new tensor with the sine of the elements of :attr:`input`.

    sin(x: Tensor) -> Tensor

    .. math::
        \text{y}_{i} = \sin(\text{x}_{i})

    Args:
        x (Tensor): the input tensor.

    For example:
    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.tensor(np.array([-0.5461,  0.1347, -2.7266, -0.2746]).astype(np.float32))
        >>> y1 = flow.sin(x1)
        >>> y1
        tensor([-0.5194,  0.1343, -0.4032, -0.2712], dtype=oneflow.float32)

        >>> x2 = flow.tensor(np.array([-1.4, 2.6, 3.7]).astype(np.float32), device=flow.device('cuda'))
        >>> y2 = flow.sin(x2)
        >>> y2
        tensor([-0.9854,  0.5155, -0.5298], device='cuda:0', dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.sin_,
    r"""sin_(x)
    In-place version of :func:`oneflow.sin`

    """,
)

add_docstr(
    oneflow.sinh,
    r"""sinh(input)
    Returns a new tensor with the hyperbolic sine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \sinh(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x1 = flow.tensor(np.array([1, 2, 3]), dtype=flow.float32)
        >>> x2 = flow.tensor(np.array([1.53123589,0.54242598,0.15117185]), dtype=flow.float32)
        >>> x3 = flow.tensor(np.array([1,0,-1]), dtype=flow.float32)

        >>> flow.sinh(x1).numpy()
        array([ 1.1752012,  3.6268604, 10.017875 ], dtype=float32)
        >>> flow.sinh(x2).numpy()
        array([2.20381  , 0.5694193, 0.1517483], dtype=float32)
        >>> flow.sinh(x3).numpy()
        array([ 1.1752012,  0.       , -1.1752012], dtype=float32)

    """,
)

add_docstr(
    oneflow.tan,
    r"""tan(input)
    Returns  the tan value of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \tan(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([-1/4*np.pi, 0, 1/4*np.pi]).astype(np.float32)
        >>> input = flow.tensor(np_arr, dtype=flow.float32)
        >>> output = flow.tan(input)
        >>> output
        tensor([-1.,  0.,  1.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.cos,
    r"""cos(input)
    Returns a new tensor with the cosine  of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \cos(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array([1.4309,  1.2706, -0.8562,  0.9796])
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> output = flow.cos(input).numpy()

    """,
)

add_docstr(
    oneflow.cosh,
    r"""cosh(input)
    Returns a new tensor with the hyperbolic cosine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \cosh(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> arr = np.array([ 0.1632,  1.1835, -0.6979, -0.7325])
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> output = flow.cosh(input).numpy()
        >>> output
        array([1.0133467, 1.7859949, 1.2535787, 1.2804903], dtype=float32)

    """,
)

add_docstr(
    oneflow.erf,
    r"""erf(x)
    Computes the error function of each element. The error function is defined as follows:

    .. math::
            \operatorname{erf}(x)=\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.tensor(np.array([0, -1., 10.]), dtype=flow.float32)
        >>> out = flow.erf(x)
        >>> out.shape
        oneflow.Size([3])
        >>> out.numpy()
        array([ 0.       , -0.8427008,  1.       ], dtype=float32)

        >>> x = flow.tensor(np.array([[0, -1., 10.], [5, 7, 0.8]]), dtype=flow.float32)
        >>> out = flow.erf(x)
        >>> out.shape
        oneflow.Size([2, 3])
        >>> out.numpy()
        array([[ 0.        , -0.8427008 ,  1.        ],
               [ 1.        ,  1.        ,  0.74210095]], dtype=float32)

        >>> x = flow.tensor(np.array([[0, -1., 10.], [5, 7, 0.8], [2, 3, 4]]), dtype=flow.float32)
        >>> out = x.erf()
        >>> out.shape
        oneflow.Size([3, 3])
        >>> out.numpy()
        array([[ 0.        , -0.8427008 ,  1.        ],
               [ 1.        ,  1.        ,  0.74210095],
               [ 0.9953223 ,  0.9999779 ,  1.        ]], dtype=float32)

    """,
)

add_docstr(
    oneflow.erfc,
    r"""erfc(x)
    Computes the complementary error function of each element of input. The complementary error
    function is defined as follows:

    .. math::
            \operatorname{erfc}(x)=1-\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.tensor(np.array([0, -1., 10.]), dtype=flow.float32)
        >>> out = flow.erfc(x)
        >>> out
        tensor([1.0000e+00, 1.8427e+00, 2.8026e-45], dtype=oneflow.float32)

        >>> x = flow.tensor(np.array([[0, -1., 10.], [5, 7, 0.8]]), dtype=flow.float32)
        >>> out = flow.erfc(x)
        >>> out
        tensor([[1.0000e+00, 1.8427e+00, 2.8026e-45],
                [1.5375e-12, 4.1838e-23, 2.5790e-01]], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.expm1,
    r"""expm1(input)
    Returns a new tensor with the exponential of the elements minus 1
    of :attr:`input`.


    The equation is:

    .. math::
        y_{i} = e^{x_{i}} - 1

    Args:
        input (oneflow.Tensor): A Tensor.

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = flow.expm1(x)
        >>> y.shape
        oneflow.Size([3])
        >>> y
        tensor([ 1.7183,  6.3891, 19.0855], dtype=oneflow.float32)

        >>> x = flow.tensor(np.array([[[2, 4, 6],[7, 8, 9]],[[10,11,12],[13,14,15]]]).astype(np.float32))
        >>> y = flow.expm1(x)
        >>> print(y.shape)
        oneflow.Size([2, 2, 3])
        >>> print(y.numpy())
        [[[6.3890562e+00 5.3598152e+01 4.0242880e+02]
          [1.0956332e+03 2.9799580e+03 8.1020840e+03]]
        <BLANKLINE>
         [[2.2025465e+04 5.9873141e+04 1.6275380e+05]
          [4.4241238e+05 1.2026032e+06 3.2690165e+06]]]


    """,
)

add_docstr(
    oneflow.fmod,
    r"""
    fmod(input, other, *, out=None) -> Tensor

    Computes the element-wise remainder of division.

    The dividend and divisor may contain both for integer and floating point
    numbers. The remainder has the same sign as the dividend :attr:`input`.

    Supports broadcasting to a common shape, integer and float inputs.


    Args:
        input (Tensor): the dividend
        other (Tensor or Scalar): the divisor

    Keyword args:
        out (Tensor, optional): the output tensor.

    Example::

        >>> import oneflow as flow
        >>> flow.fmod(flow.tensor([-3., -2, -1, 1, 2, 3], dtype=flow.float32), 2.)
        tensor([-1., -0., -1.,  1.,  0.,  1.], dtype=oneflow.float32)
        >>> flow.fmod(flow.tensor([1, 2, 3, 4, 5.], dtype=flow.float32), 1.5)
        tensor([1.0000, 0.5000, 0.0000, 1.0000, 0.5000], dtype=oneflow.float32)
        >>> flow.fmod(flow.tensor([1, 2, 3, 4., -5]), flow.tensor([4, 2, 1, 3., 1]))
        tensor([1., 0., 0., 1., -0.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.log,
    r"""log(input)
    Returns a new tensor with the natural logarithm of the elements of :attr:`input`.

    .. math::
        y_{i} = \log_{e} (x_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.random.randn(2, 3, 4, 5)
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> output = flow.log(input)


    """,
)

add_docstr(
    oneflow.log2,
    """
    oneflow.log2(input) -> Tensor

    Returns a new tensor with the natural logarithm to the base 2 of the elements of :attr:`input`.
    
    .. math::
        y_{i} = \\log2_{e} (x_{i})

    Args:
        input (Tensor): the input tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.random.randn(2, 3, 4, 5)
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> output = flow.log2(input)


    """,
)

add_docstr(
    oneflow.minimum,
    r"""Computes the element-wise minimum of x and y.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor((1, 2, -1), dtype=flow.float32)
        >>> y = flow.tensor((3, 0, 4), dtype=flow.float32)
        >>> flow.minimum(x, y)
        tensor([ 1.,  0., -1.], dtype=oneflow.float32)

        >>> x = flow.tensor((1,), dtype=flow.float32)
        >>> y = flow.tensor((3, 0, 4), dtype=flow.float32)
        >>> flow.minimum(x, y)
        tensor([1., 0., 1.], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.maximum,
    r"""Computes the element-wise maximum of x and y.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor((1, 2, -1), dtype=flow.float32)
        >>> y = flow.tensor((3, 0, 4), dtype=flow.float32)
        >>> flow.maximum(x, y)
        tensor([3., 2., 4.], dtype=oneflow.float32)

        >>> x = flow.tensor((1,), dtype=flow.float32)
        >>> y = flow.tensor((3, 0, 4), dtype=flow.float32)
        >>> flow.maximum(x, y)
        tensor([3., 1., 4.], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.pow,
    r"""pow(input, exponent)
    Takes the power of each element in input with exponent and returns a tensor with the result. Exponent can be either a single float number, a single int number, or a tensor with the same shape as input.
    When exponent is a scalar value, the operation applied is:

    .. math::
        \\text{out}_i = x_i ^ \\text{exponent}
\u200b
    When exponent is a tensor, the operation applied is:

    .. math::
        \\text{out}_i = x_i ^ {\\text{exponent}_i}

    Args:
        - input (Tensor): the input tensor.
        - exponent (int, float, Tensor): the exponent.

    Returns:
        Tensor: The result of variance on the specified axis of input Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), dtype=flow.float32)
        >>> out = flow.pow(x, 2)
        >>> out
        tensor([ 1.,  4.,  9., 16., 25., 36.], dtype=oneflow.float32)

        >>> x = flow.tensor(np.array([1.0, 2.0, 3.0, 4.0]), dtype=flow.float32)
        >>> y = flow.tensor(np.array([1.0, 2.0, 3.0, 4.0]), dtype=flow.float32)
        >>> out = flow.pow(x, y)
        >>> out
        tensor([  1.,   4.,  27., 256.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.rsqrt,
    r"""rsqrt(input)
    Returns a new tensor with the reciprocal of the square-root of each of
        the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np

            >>> a = flow.tensor(np.array([1.0, 2.0, 3.0]), dtype=flow.float32)
            >>> out = flow.rsqrt(a).numpy()
            >>> out
            array([1.        , 0.70710677, 0.57735026], dtype=float32)
    """,
)

add_docstr(
    oneflow.sqrt,
    r"""sqrt(input)
    Returns a new tensor with the square-root of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \sqrt{\text{input}_{i}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np

            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> input = flow.tensor(arr, dtype=flow.float32)
            >>> output = flow.sqrt(input).numpy()
            >>> output
            array([1.       , 1.4142135, 1.7320508], dtype=float32)
        """,
)


add_docstr(
    oneflow.square,
    r"""square(input)
    Returns a new tensor with the square of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \sqrt{\text{input}_{i}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np

            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> input = flow.tensor(arr, dtype=flow.float32)
            >>> output = flow.square(input).numpy()
            >>> output
            array([1., 4., 9.], dtype=float32)
        """,
)

add_docstr(
    oneflow.matmul,
    r"""
    matmul(input, other) -> Tensor

    This operator applies matrix multiplication to two Tensor.

    Args:
        a (oneflow.Tensor): A Tensor
        b (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input1 = flow.tensor(np.random.randn(2, 6), dtype=flow.float32)
        >>> input2 = flow.tensor(np.random.randn(6, 5), dtype=flow.float32)
        >>> of_out = flow.matmul(input1, input2)
        >>> of_out.shape
        oneflow.Size([2, 5])

    """,
)

add_docstr(
    oneflow.round,
    r"""round(input)
    This operator rounds the value of Blob to the nearest integer.
    Args:
        input (oneflow.Tensor): A Tensor
    Returns:
        oneflow.Tensor: The result Tensor
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.tensor(np.array([1.49999, 1.500001, 2.7]).astype(np.float32))
        >>> out1 = flow.round(x1)
        >>> out1.numpy()
        array([1., 2., 3.], dtype=float32)
        >>> x2 = flow.tensor(np.array([2.499999, 7.5000001, 5.3, 6.8]).astype(np.float32))
        >>> out2 = flow.round(x2)
        >>> out2.numpy()
        array([2., 8., 5., 7.], dtype=float32)

    """,
)

add_docstr(
    oneflow.std,
    r"""std(input, dim, unbiased, keepdim)
    Returns the standard-deviation of each row of the :attr:`input` tensor in the
    dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them.

    If keepdim is True, the output tensor is of the same size as input except in
    the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed,
    resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).

    If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
    via the biased estimator. Otherwise, Bessel's correction will be used.

    Args:
        input (Tensor): the input tensor.
        dim (int or tuple of python:ints): the dimension or dimensions to reduce.
        unbiased (bool): whether to use the unbiased estimation or not
        keepdim (bool): whether the output tensor has `dim` retained or not.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> input = flow.tensor(arr)
        >>> output = flow.std(input, dim=0).numpy()
        >>> output
        array(1.)

    """,
)

add_docstr(
    oneflow.var,
    r"""var(input, dim, unbiased, keepdim)
    Returns the variance of each row of the `input` tensor in the given dimension `dim`.

    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim`
    where it is of size 1. Otherwise, dim is squeezed (see `flow.squeeze()`), resulting in the output
    tensor having 1 (or `len(dim)`) fewer dimension(s).

    Args:
        input (Tensor): the input tensor.
        dim (int or tuple of python:ints): the dimension or dimensions to reduce. Defaults to None.
        unbiased (bool, optional): whether to use Bessel’s correction (:math:`\delta N = 1`). Defaults to True.
        keepdim (bool, optional): whether the output tensor has dim retained or not. Defaults to False.

    Returns:
        Tensor: The result of variance on the specified axis of input Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.tensor(np.random.randn(2, 3, 4, 5))
        >>> output = flow.var(input, 1, True)

    """,
)

add_docstr(
    oneflow.logical_not,
    r"""logical_not(input, other)
    Computes the element-wise logical NOT of the given input tensors.
    Zeros are treated as False and nonzeros are treated as True.
    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute NOT with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1, 0, -1], dtype=flow.float32)
        >>> out = flow.logical_not(input)
        >>> out
        tensor([False,  True, False], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.dot,
    r"""dot(input, other)
    This operator computes the dot product of tensor input and other.

    The equation is:

	$$
        \\sum_{i=1}^{n}(x[i] * y[i])
	$$

    Args:
        input (Tensor):  first tensor in the dot product.
        other (Tensor):  second tensor in the dot product.

    Shape:
        - input: Input must be 1D.
        - other: Other must be 1D.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.dot(flow.Tensor([2, 3]), flow.Tensor([2, 1]))
        tensor(7., dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.select,
    r"""select(input, dim, select)
    Slices the self tensor along the selected dimension at the given index. This function returns 
    a view of the original tensor with the given dimension removed.

    Args:
        input (Tensor): the input tensor.
        dim  (int):  the dimension to slice.
        select (int): the index to select with.

    Returns:
        oneflow.Tensor: the output Tensor.

    For example:
    
    .. code-block:: python
    
        >>> import oneflow as flow
        >>> input = flow.rand(3, 4, 5)
        >>> out = flow.select(input, 0, 1)
        >>> out.size()
        oneflow.Size([4, 5])
        >>> out = flow.select(input, 1, 1)
        >>> out.size()
        oneflow.Size([3, 5])
    """,
)

add_docstr(
    oneflow.movedim,
    r"""movedim(input, source, destination)
    Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.
    Other dimensions of input that are not explicitly moved remain in their original order and appear at the positions not specified in destination.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.movedim.html#torch.movedim.

    Args:
        input (Tensor): the input tensor.
        source  (int or a list): Original positions of the dims to move. These must be unique.
        destination (int or a list): Destination positions for each of the original dims. These must also be unique.

    Returns:
        oneflow.Tensor: the output Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        >>> output = flow.movedim(input, 1, 0)
        >>> output.shape
        oneflow.Size([3, 2, 4, 5])
        >>> output = flow.movedim(input, (1, 2), (0, 1))
        >>> output.shape
        oneflow.Size([3, 4, 2, 5])
    """,
)

add_docstr(
    oneflow.as_strided,
    r"""
    Create a view of an existing oneflow.Tensor input with specified size, stride and storage_offset.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.as_strided.html#torch.as_strided.

    Args:
        input (Tensor): the input tensor.
        size (tuple or ints): the shape of the output tensor.
        stride (tuple or ints): the stride of the output tensor.
        storage_offset (int): the offset in the underlying storage of the output tensor

    Returns:
        oneflow.Tensor: the output Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.rand(2,3,5)
        >>> output = flow.as_strided(input, (2,3,3), (1,2,3), 1)
        >>> output.size()
        oneflow.Size([2, 3, 3])
    """,
)

add_docstr(
    oneflow.eye,
    """oneflow.eye(n, m, *, device=None, requires_grad=False, placement=None, sbp) -> Tensor

    This operator creates a 2-D Tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): the number of rows.
        m (int, optional): the number of colums with default being n. Defaults to None.

    Keyword args:
        device(Union[flow.device, str], optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: `False`.
        placement(oneflow._oneflow_internal.placement, optional): The placement attribute allows you to specify which physical device the tensor is stored on.
        sbp(Union[oneflow._oneflow_internal.sbp.sbp, List[oneflow._oneflow_internal.sbp.sbp]], optional): When creating a global tensor, specify the SBP of the tensor.

    Returns:
        oneflow.Tensor: The result tensor with ones on the diagonal and zeros elsewhere.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> out = flow.eye(3, 3)
        >>> out
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]], dtype=oneflow.float32)
        >>> out = flow.eye(3, 3, device="cuda")
        >>> out
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]], device='cuda:0', dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.tensor_split,
    r"""tensor_split(input, indices_or_sections, dim)
    Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension
    dim according to the indices or number of sections specified by indices_or_sections .
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.tensor_split.html#torch.tensor_split

    Args:
        input (Tensor): the input tensor.
        indices_or_sections (int or a list): If indices_or_sections is an integer n , input is split into n sections 
            along dimension dim.If input is divisible by n along dimension dim, each section will be of equal size, 
            input.size (dim) / n. If input is not divisible by n, the sizes of the first int(input.size(dim) % n).
            sections will have size int(input.size(dim) / n) + 1, and the rest will have size int(input.size(dim) / n).
            If indices_or_sections is a list or tuple of ints, then input is split along dimension dim at each of the indices in 
            the list, tuple or tensor. For instance, indices_or_sections=[2, 3] and dim=0 would result in the tensors 
            input[:2], input[2:3], and input[3:].If indices_or_sections is a tensor, it must be a zero-dimensional or
            one-dimensional long tensor on the CPU.
        dim (int): dimension along which to split the tensor.

    Returns:
        oneflow.TensorTuple: the output TensorTuple.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.rand(3,4,5)
        >>> output = flow.tensor_split(input,(2,3),2)
        >>> output[0].size()
        oneflow.Size([3, 4, 2])
        >>> output[1].size()
        oneflow.Size([3, 4, 1])
        >>> output[2].size()
        oneflow.Size([3, 4, 2])
    """,
)

add_docstr(
    oneflow.hsplit,
    r"""hsplit(input, indices_or_sections)
    Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections.
    Each split is a view of input.
    If input is one dimensional this is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=0) 
    (the split dimension is zero), and if input has two or more dimensions it’s equivalent to calling 
    torch.tensor_split(input, indices_or_sections, dim=1) (the split dimension is 1), except that if indices_or_sections
    is an integer it must evenly divide the split dimension or a runtime error will be thrown.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.hsplit.html#torch.hsplit

    Args:
        input (Tensor): the input tensor.
        indices_or_sections (int or a list): If indices_or_sections is an integer n , input is split into n sections 
            along dimension dim.If input is divisible by n along dimension dim, each section will be of equal size, 
            input.size (dim) / n. If input is not divisible by n, the sizes of the first int(input.size(dim) % n).
            sections will have size int(input.size(dim) / n) + 1, and the rest will have size int(input.size(dim) / n).
            If indices_or_sections is a list or tuple of ints, then input is split along dimension dim at each of the indices in 
            the list, tuple or tensor. For instance, indices_or_sections=[2, 3] and dim=0 would result in the tensors 
            input[:2], input[2:3], and input[3:].If indices_or_sections is a tensor, it must be a zero-dimensional or
            one-dimensional long tensor on the CPU.

    Returns:
        oneflow.TensorTuple: the output TensorTuple.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.rand(3,4,5,6)
        >>> output = flow.hsplit(input,(1,3))
        >>> output[0].size()
        oneflow.Size([3, 1, 5, 6])
        >>> output[1].size()
        oneflow.Size([3, 2, 5, 6])
        >>> output[2].size()
        oneflow.Size([3, 1, 5, 6])
    """,
)

add_docstr(
    oneflow.vsplit,
    r"""vsplit(input, indices_or_sections)
    Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections.
    Each split is a view of input.
    This is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=0) (the split dimension is 0),
    except that if indices_or_sections is an integer it must evenly divide the split dimension or a runtime error will be thrown.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.vsplit.html#torch.vsplit

    Args:
        input (Tensor): the input tensor.
        indices_or_sections (int or a list): If indices_or_sections is an integer n , input is split into n sections 
            along dimension dim.If input is divisible by n along dimension dim, each section will be of equal size, 
            input.size (dim) / n. If input is not divisible by n, the sizes of the first int(input.size(dim) % n).
            sections will have size int(input.size(dim) / n) + 1, and the rest will have size int(input.size(dim) / n).
            If indices_or_sections is a list or tuple of ints, then input is split along dimension dim at each of the indices in 
            the list, tuple or tensor. For instance, indices_or_sections=[2, 3] and dim=0 would result in the tensors 
            input[:2], input[2:3], and input[3:].If indices_or_sections is a tensor, it must be a zero-dimensional or
            one-dimensional long tensor on the CPU.

    Returns:
        oneflow.TensorTuple: the output TensorTuple.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.rand(3,4,5,6)
        >>> output = flow.vsplit(input,(1,3))
        >>> output[0].size()
        oneflow.Size([1, 4, 5, 6])
        >>> output[1].size()
        oneflow.Size([2, 4, 5, 6])
        >>> output[2].size()
        oneflow.Size([1, 4, 5, 6])
    """,
)

add_docstr(
    oneflow.cumsum,
    r"""oneflow.cumsum(input, dim) -> Tensor
    
    This operator computes the cumulative sum of input elements in the given dimension.

    The equation is:

	$$
        y_{i}=x_{0}+x_{1}+...+x_{i}
	$$

    Args:
        input (Tensor):  the input ND tensor.
        dim (int):  the dimension to do cumsum, valid range is [-N, N-1), N is tensor's dimensions

    Returns:
        oneflow.Tensor: The result tensor with cumsum result.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.ones(3, 3)
        >>> dim = 1
        >>> flow.cumsum(input, dim)
        tensor([[1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.cumprod,
    """oneflow.cumprod(input, dim) -> Tensor

    This operator computes the cumulative product of input elements in the given dimension.

    The equation is:

	$$
        y_{i}=x_{0}*x_{1}*...*x_{i}
	$$

    Args:
        input (Tensor):  the input tensor.
        dim (int):  the dimension to do cumsum whose valid range is [-N, N-1), and the N is tensor's dimensions

    Returns:
        oneflow.Tensor: The result tensor with cumprod result.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input=flow.tensor([1, 2, 3])
        >>> flow.cumprod(input, dim=0)
        tensor([1, 2, 6], dtype=oneflow.int64)
    """,
)
