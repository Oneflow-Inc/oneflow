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
import collections
from typing import Optional, Sequence, Union

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _check_axis
from oneflow.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


@register_tensor_op("sub")
def _sub(input, other):
    """Computes the subtraction of input by other for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = input - other
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise subtract
        >>> input = flow.Tensor(np.random.randn(2,3))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar subtract
        >>> input = 5
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast subtract
        >>> input = flow.Tensor(np.random.randn(1,1))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

    """
    return flow._C.sub(input, other)


@register_tensor_op("div")
def _div(input, other):
    """Computes the division of input by other for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = \\frac{input}{other}
    
    Args:
        input (Union[int, float, flow.Tensor]): input.
        other (Union[int, float, flow.Tensor]): other.
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise divide
        >>> input = flow.Tensor(np.random.randn(2,3))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar divide
        >>> input = 5
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast divide
        >>> input = flow.Tensor(np.random.randn(1,1))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape 
        (2, 3)

    """
    return flow._C.div(input, other)


@register_tensor_op("reciprocal")
def _reciprocal(x):
    """Computes the safe reciprocal of x. If x is zero, the reciprocal will
    be also set to zero.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        >>> out = flow.reciprocal(x)
        >>> out.numpy()
        array([[1.        , 0.5       , 0.33333334],
               [0.25      , 0.2       , 0.16666667]], dtype=float32)
    """
    return flow._C.reciprocal_no_nan(x)


@register_tensor_op("add")
def _add(input, other):
    """Computes the addition of `input` by `other` for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = input + other

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise add
        >>> x = flow.Tensor(np.random.randn(2,3))
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # scalar add
        >>> x = 5
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # broadcast add
        >>> x = flow.Tensor(np.random.randn(1,1))
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

    """
    return flow._C.add(input, other)


@register_tensor_op("add_")
def _add_inplace(x, y):
    """
    In-place version of :func:`oneflow.Tensor.add`.
    """
    return flow._C.add(x, y, inplace=True)


def asin_op(input):
    """
    Returns a new tensor with the arcsine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\sin^{-1}(\\text{input}_{i})

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
    """
    return flow._C.asin(input)


@register_tensor_op("asin")
def asin_op_tensor(input):
    """

    See :func:`oneflow.asin`
    """
    return flow._C.asin(input)


def arcsin_op(input):
    """
  
    Alias for :func:`oneflow.asin`
    """
    return flow._C.asin(input)


@register_tensor_op("arcsin")
def arcsin_op_tensor(input):
    """

    See :func:`oneflow.asin`
    """
    return flow._C.asin(input)


def asinh_op(input):
    """
    Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\sinh^{-1}(\\text{input}_{i})

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

    """
    return flow._C.asinh(input)


def arcsinh_op(input):
    """
  
    Alias for :func:`oneflow.asinh`
    """
    return flow._C.asinh(input)


@register_tensor_op("asinh")
def asinh_op_tensor(input):
    """

    See :func:`oneflow.asinh`
    """
    return flow._C.asinh(input)


@register_tensor_op("arcsinh")
def arcsinh_op_tensor(input):
    """

    See :func:`oneflow.asinh`
    """
    return flow._C.asinh(input)


@register_tensor_op("sin")
def sin_op_tensor(input):
    """

    sin() -> Tensor

    See :func:`oneflow.sin`
    
    """
    return flow._C.sin(input)


@register_tensor_op("sin_")
def inplace_sin_op_tensor(input):
    """
    In-place version of :func:`oneflow.sin`
    
    """
    return flow._C.sin_(input)


@register_tensor_op("cos")
def cos_op(input):
    """
    Returns a new tensor with the cosine  of the elements of :attr:`input`.
    
    .. math::
        \\text{out}_{i} = \\cos(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array([1.4309,  1.2706, -0.8562,  0.9796])
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> output = flow.cos(input).numpy()

    """
    return flow._C.cos(input)


def atan_op(input):
    """
    Returns a new tensor with the arctangent of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\tan^{-1}(\\text{input}_{i})

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
        
    """
    return flow._C.atan(input)


@register_tensor_op("atan")
def atan_op_tensor(input):
    """

    See :func:`oneflow.atan`
    
    """
    return flow._C.atan(input)


def arctan_op(input):
    """
    Alias for :func:`oneflow.atan`
    
    """
    return flow._C.atan(input)


@register_tensor_op("arctan")
def arctan_op_tensor(input):
    """

    See :func:`oneflow.arctan`
    
    """
    return flow._C.atan(input)


def fmod_op(input, other):
    """
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
        >>> flow.fmod(flow.tensor([-3., -2, -1, 1, 2, 3]), 2.)
        tensor([-1., -0., -1.,  1.,  0.,  1.], dtype=oneflow.float32)
        >>> flow.fmod(flow.tensor([1, 2, 3, 4, 5.]), 1.5)
        tensor([1.0000, 0.5000, 0.0000, 1.0000, 0.5000], dtype=oneflow.float32)
        >>> flow.fmod(flow.tensor([1, 2, 3, 4., -5]), flow.tensor([4, 2, 1, 3., 1]))
        tensor([1., 0., 0., 1., -0.], dtype=oneflow.float32)

    """
    return flow._C.fmod(input, other)


@register_tensor_op("fmod")
def fmod_op_tensor(input, other):
    """

    See :func:`oneflow.fmod`
    
    """
    return fmod_op(input, other)


@register_tensor_op("log")
def log_op(input):
    """
    Returns a new tensor with the natural logarithm of the elements of :attr:`input`.
    
    .. math::
        y_{i} = \\log_{e} (x_{i})

    Args:
        input (Tensor): the input tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.random.randn(2, 3, 4, 5)
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> output = flow.log(input)


    """
    return flow._C.log(input)


@register_tensor_op("log2")
def log2_op(input):
    """
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


    """
    return flow._C.log2(input)


@register_tensor_op("rsqrt")
def rsqrt_op(input):
    """Returns a new tensor with the reciprocal of the square-root of each of
        the elements of :attr:`input`.

        .. math::
            \\text{out}_{i} = \\frac{1}{\\sqrt{\\text{input}_{i}}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np
            
            >>> a = flow.Tensor(np.array([1.0, 2.0, 3.0]))
            >>> out = flow.rsqrt(a).numpy()
            >>> out
            array([1.        , 0.70710677, 0.57735026], dtype=float32)
    """
    return flow._C.rsqrt(input)


@register_tensor_op("sqrt")
def sqrt_op(input):
    """Returns a new tensor with the square-root of the elements of :attr:`input`.

        .. math::
            \\text{out}_{i} = \\sqrt{\\text{input}_{i}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np
            
            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> input = flow.Tensor(arr)
            >>> output = flow.sqrt(input).numpy()
            >>> output
            array([1.       , 1.4142135, 1.7320508], dtype=float32)
        """
    return flow._C.sqrt(input)


@register_tensor_op("square")
def square_op(input):
    """Returns a new tensor with the square of the elements of :attr:`input`.

        .. math::
            \\text{out}_{i} = \\sqrt{\\text{input}_{i}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np
            
            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> input = flow.Tensor(arr)
            >>> output = flow.square(input).numpy()
            >>> output
            array([1., 4., 9.], dtype=float32)
        """
    return flow._C.square(input)


def addmm(x, mat1, mat2, alpha=1, beta=1):
    if len(x.shape) > 2 or len(mat1.shape) > 2 or len(mat2.shape) > 2:
        raise ValueError("input matrixes shape can not be greater than 2")
    else:
        return flow.mul(x, beta) + flow.mul(flow._C.matmul(mat1, mat2), alpha)


def addmm_op(input, mat1, mat2, alpha=1, beta=1):
    """addmm(beta=1, input, alpha=1, mat1, mat2, out=None) -> Tensor

    Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
    The matrix :attr:`input` is added to the final result.

    If :attr:`mat1` is a :math:`(n \\times m)` tensor, :attr:`mat2` is a
    :math:`(m \\times p)` tensor, then :attr:`input` must be
    broadcastable with a :math:`(n \\times p)` tensor
    and :attr:`out` will be a :math:`(n \\times p)` tensor.

    :attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
    :attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.

    .. math::
        \\text{out} = \\beta\\ \\text{input} + \\alpha\\ (\\text{mat1}_i \\mathbin{@} \\text{mat2}_i)

    For inputs of type `float` or `double`, arguments :attr:`beta` and
    :attr:`alpha` must be real numbers, otherwise they should be integers.

    Args:
        beta (Number, optional): multiplier for :attr:`input` (:math:`\\beta`)
        input (Tensor): matrix to be added
        alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\\alpha`)
        mat1 (Tensor): the first matrix to be multiplied
        mat2 (Tensor): the second matrix to be multiplied
        out (Tensor, optional): the output tensor.

    For example:

        >>> import numpy as np
        >>> import oneflow as flow
        >>> input = flow.tensor(np.array([[1,2,4],[5,11,9.1]]))
        >>> mat1 = flow.tensor(np.array([[7.3,1.9,7.3],[10.2,1,5.5]])) 
        >>> mat2 = flow.tensor(np.array([[7.3,1.9,7.3],[10.2,1,5.5],[3.7,2.2,8.1]])) 
        >>> output = flow.addmm(input, mat1, mat2)
        >>> output
        tensor([[100.6800,  33.8300, 126.8700],
                [110.0100,  43.4800, 133.6100]], dtype=oneflow.float64)
        >>> output.shape
        oneflow.Size([2, 3])

        >>> input2 = flow.tensor(np.array([1.7]))
        >>> mat1 = flow.tensor(np.array([[1,2],[5,9.1],[7.7,1.4]]))
        >>> mat2 = flow.tensor(np.array([[1,2,3.7],[5,9.1,6.8]]))
        >>> output2 = flow.addmm(input2, mat1, mat2, alpha=1, beta=2)
        >>> output2
        tensor([[14.4000, 23.6000, 20.7000],
                [53.9000, 96.2100, 83.7800],
                [18.1000, 31.5400, 41.4100]], dtype=oneflow.float64)
        >>> output2.shape
        oneflow.Size([3, 3])
    """
    return addmm(input, mat1, mat2, alpha, beta)


@register_tensor_op("addmm")
def addmm_op_tensor(input, mat1, mat2, alpha=1, beta=1):
    """
    See :func:`oneflow.addmm`
    """
    return addmm(input, mat1, mat2, alpha, beta)


@register_tensor_op("cosh")
def cosh_op(input):
    """
    Returns a new tensor with the hyperbolic cosine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\cosh(\\text{input}_{i})

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

    """
    return flow._C.cosh(input)


@register_tensor_op("erf")
def erf_op(input):
    """Computes the error function of each element. The error function is defined as follows:

    .. math::
            \\operatorname{erf}(x)=\\frac{2}{\\sqrt{\\pi}} \\int_{0}^{x} e^{-t^{2}} d t

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

    """
    return flow._C.erf(input)


@register_tensor_op("erf")
def erf_op_tensor(input):
    """
    See :func:`oneflow.erf`
    """
    return flow._C.erf(input)


@register_tensor_op("erfc")
def erfc_op(input):
    """Computes the complementary error function of each element of input. The complementary error 
    function is defined as follows:

    .. math::
            \\operatorname{erfc}(x)=1-\\frac{2}{\\sqrt{\\pi}} \\int_{0}^{x} e^{-t^{2}} d t

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
        
    """
    return flow._C.erfc(input)


@register_tensor_op("erfc")
def erfc_op_tensor(input):
    """
    See :func:`oneflow.erfc`
    """
    return flow._C.erfc(input)


def ceil_op(input):
    """Returns a new tensor with the ceil of the elements of :attr:`input`,
    the smallest integer greater than or equal to each element.

    The equation is: 

    .. math::
        \\text{out}_{i} = \\left\\lceil \\text{input}_{i} \\right\\rceil = \\left\\lfloor \\text{input}_{i} \\right\\rfloor + 1

    Args:
        input (oneflow.Tensor): A Tensor.
    
    Returns:
        oneflow.Tensor: The result Tensor

    For example: 


    .. code-block:: python 
        
        >>> import oneflow as flow
        >>> import numpy as np   
        >>> x = flow.Tensor(np.array([0.1, -2, 3.4]).astype(np.float32))
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([3])
        >>> y
        tensor([ 1., -2.,  4.], dtype=oneflow.float32)
        >>> x = flow.Tensor(np.array([[2.5, 4.6, 0.6],[7.8, 8.3, 9.2]]).astype(np.float32))
        >>> y = x.ceil()
        >>> y.shape
        oneflow.Size([2, 3])
        >>> y
        tensor([[ 3.,  5.,  1.],
                [ 8.,  9., 10.]], dtype=oneflow.float32)
        >>> x = flow.Tensor(np.array([[[2.2, 4.4, 6.5],[7.1, 8.2, 9.3]],[[10.6,11.2,12.2],[13.5,14.8,15.9]]]).astype(np.float32))
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([2, 2, 3])
        >>> y
        tensor([[[ 3.,  5.,  7.],
                 [ 8.,  9., 10.]],
        <BLANKLINE>
                [[11., 12., 13.],
                 [14., 15., 16.]]], dtype=oneflow.float32)

    """
    return flow._C.ceil(input)


@register_tensor_op("ceil")
def ceil_op_tensor(input):
    """
    See :func:`oneflow.ceil`
    """
    return flow._C.ceil(input)


def expm1_op(input):
    """Returns a new tensor with the exponential of the elements minus 1
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
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = flow.expm1(x)
        >>> y.shape
        oneflow.Size([3])
        >>> y
        tensor([ 1.7183,  6.3891, 19.0855], dtype=oneflow.float32)


        >>> x = flow.Tensor(np.array([[2, 4, 6],[7, 8, 9]]).astype(np.float32))
        >>> y = x.expm1()
        >>> y.shape
        oneflow.Size([2, 3])
        >>> y
        tensor([[6.3891e+00, 5.3598e+01, 4.0243e+02],
                [1.0956e+03, 2.9800e+03, 8.1021e+03]], dtype=oneflow.float32)



        >>> x = flow.Tensor(np.array([[[2, 4, 6],[7, 8, 9]],[[10,11,12],[13,14,15]]]).astype(np.float32))
        >>> y = flow.expm1(x)
        >>> print(y.shape)
        oneflow.Size([2, 2, 3])
        >>> print(y.numpy())
        [[[6.3890562e+00 5.3598152e+01 4.0242880e+02]
          [1.0956332e+03 2.9799580e+03 8.1020840e+03]]
        <BLANKLINE>
         [[2.2025465e+04 5.9873141e+04 1.6275380e+05]
          [4.4241238e+05 1.2026032e+06 3.2690165e+06]]]


    """
    return flow._C.expm1(input)


@register_tensor_op("expm1")
def expm1_op_tensor(input):
    """
    See :func:`oneflow.expm1`
    """
    return flow._C.expm1(input)


class Topk(Module):
    def __init__(
        self, k, dim: int = None, largest: bool = True, sorted: bool = True
    ) -> None:
        super().__init__()
        self.k = k
        self.sorted = sorted
        self.dim = dim
        self.largest = largest

    def forward(self, input):
        if self.dim == None:
            self.dim = -1
        num_axes = len(input.shape)
        axis = self.dim if self.dim >= 0 else self.dim + num_axes
        assert 0 <= axis < num_axes, "axis out of range"
        if axis == num_axes - 1:
            if self.largest:
                indices = flow._C.top_k(input, self.k)
            else:
                neg_input = flow.mul(input, -1)
                indices = flow._C.top_k(neg_input, self.k)
            return (flow.gather(input, axis, indices), indices)
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
            x = flow._C.transpose(input, perm=perm)
            if self.largest:
                indices = flow._C.top_k(x, self.k)
            else:
                neg_input = flow.mul(x, -1)
                indices = flow._C.top_k(neg_input, self.k)
            indices = flow._C.transpose(indices, perm=get_inversed_perm(perm))
            return (flow.gather(input, axis, indices), indices)


@register_tensor_op("topk")
def topk_op(input, k, dim: int = None, largest: bool = True, sorted: bool = True):
    """Finds the values and indices of the k largest entries at specified axis.

    Args:
        input (oneflow.Tensor): Input Tensor
        k (int): the k in “top-k”
        dim (int, optional): the dimension to sort along. Defaults to the last dim (-1)
        largest (bool, optional): controls whether to return largest or smallest elements
        sorted (bool, optional): controls whether to return the elements in sorted order (Only Support True Now!)

    Returns:
        Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int32)): A tuple of (values, indices), where
        the indices are the indices of the elements in the original input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[1, 3, 8, 7, 2], [1, 9, 4, 3, 2]], dtype=np.float32)
        >>> (values, indices) = flow.topk(flow.Tensor(x), k=3, dim=1)
        >>> values
        tensor([[8., 7., 3.],
                [9., 4., 3.]], dtype=oneflow.float32)
        >>> indices
        tensor([[2, 3, 1],
                [1, 2, 3]], dtype=oneflow.int64)
        >>> values.shape
        oneflow.Size([2, 3])
        >>> indices.shape
        oneflow.Size([2, 3])
        >>> (values, indices) = flow.topk(flow.Tensor(x), k=2, dim=1, largest=False)
        >>> values
        tensor([[1., 2.],
                [1., 2.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 4],
                [0, 4]], dtype=oneflow.int64)
        >>> values.shape
        oneflow.Size([2, 2])
        >>> indices.shape
        oneflow.Size([2, 2])

    """
    return Topk(k=k, dim=dim, largest=largest, sorted=sorted)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
