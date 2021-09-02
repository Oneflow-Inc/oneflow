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
<<<<<<< HEAD
    oneflow.Tensor.abs, 
    """
    abs() -> Tensor
    See :func:`oneflow.abs`
    """
)

add_docstr(
    oneflow.Tensor.exp, 
    """
    exp() -> Tensor
    See :func:`oneflow.exp`
    """
)

add_docstr(
    oneflow.Tensor.acos, 
    """
    acos() -> Tensor
    See :func:`oneflow.acos`
    """
)

add_docstr(
    oneflow.Tensor.acosh, 
    """
    acosh() -> Tensor
    See :func:`oneflow.acosh`
    """
)

add_docstr(
    oneflow.Tensor.arccosh, 
    """
    arccosh() -> Tensor
    See :func:`oneflow.arccosh`
    """
)

add_docstr(
    oneflow.Tensor.atanh, 
    """
    atanh() -> Tensor
    See :func:`oneflow.atanh`
    """
)

add_docstr(
    oneflow.Tensor.arctanh, 
    """
    arctanh() -> Tensor
    See :func:`oneflow.arctanh`
    """
)

add_docstr(
    oneflow.Tensor.sign, 
    """
    sign() -> Tensor
    See :func:`oneflow.sign`
    """
)

add_docstr(
    oneflow.Tensor.sinh, 
    """
    sinh() -> Tensor
    See :func:`oneflow.sinh`
    """
)

add_docstr(
    oneflow.Tensor.tan, 
    """
    tan() -> Tensor
    See :func:`oneflow.tan`
    """
)

add_docstr(
    oneflow.Tensor.gt, 
    """
    gt() -> Tensor
    See :func:`oneflow.gt`
    """
)

add_docstr(
    oneflow.Tensor.ge, 
    """
    ge() -> Tensor
    See :func:`oneflow.ge`
    """
)

add_docstr(
    oneflow.Tensor.gelu, 
    """
    gelu() -> Tensor
    See :func:`oneflow.gelu`
    """
)

add_docstr(
    oneflow.Tensor.mish, 
    """
    mish() -> Tensor
    See :func:`oneflow.mish`
    """
)

add_docstr(
    oneflow.Tensor.sigmoid, 
    """
    sigmoid() -> Tensor
    See :func:`oneflow.sigmoid`
    """
)

add_docstr(
    oneflow.Tensor.tanh, 
    """
    tanh() -> Tensor
    See :func:`oneflow.tanh`
    """
)

add_docstr(
    oneflow.Tensor.silu, 
    """
    silu() -> Tensor
    See :func:`oneflow.silu`
    """
)

add_docstr(
    oneflow.Tensor.selu, 
    """
    selu() -> Tensor
    See :func:`oneflow.selu`
    """
)

add_docstr(
    oneflow.Tensor.softsign, 
    """
    softsign() -> Tensor
    See :func:`oneflow.softsign`
    """
)

add_docstr(
    oneflow.Tensor.cast, 
    """
    cast() -> Tensor
    See :func:`oneflow.cast`
    """
)

add_docstr(
    oneflow.Tensor.log1p, 
    """
    log1p() -> Tensor
    See :func:`oneflow.log1p`
    """
)

add_docstr(
    oneflow.Tensor.add, 
    """
    add() -> Tensor
    See :func:`oneflow.add`
    """
)

add_docstr(
    oneflow.Tensor.add_, 
    """
    add_() -> Tensor
    See :func:`oneflow.add_`
    """
)

add_docstr(
    oneflow.Tensor.div, 
    """
    div() -> Tensor
    See :func:`oneflow.div`
    """
)

add_docstr(
    oneflow.Tensor.mul, 
    """
    mul() -> Tensor
    See :func:`oneflow.mul`
    """
)


add_docstr(
    oneflow.Tensor.reciprocal, 
    """
    reciprocal() -> Tensor
    See :func:`oneflow.reciprocal`
    """
)

add_docstr(
    oneflow.Tensor.sub, 
    """
    sub() -> Tensor
    See :func:`oneflow.sub`
    """
)

add_docstr(
    oneflow.Tensor.asin, 
    """
    asin() -> Tensor
    See :func:`oneflow.asin`
    """
)

add_docstr(
    oneflow.Tensor.arcsin, 
    """
    arcsin() -> Tensor
    See :func:`oneflow.arcsin`
    """
)

add_docstr(
    oneflow.Tensor.asinh, 
    """
    asinh() -> Tensor
    See :func:`oneflow.asinh`
    """
)

add_docstr(
    oneflow.Tensor.arcsinh, 
    """
    arcsinh() -> Tensor
    See :func:`oneflow.arcsinh`
    """
)

add_docstr(
    oneflow.Tensor.atan, 
    """
    atan() -> Tensor
    See :func:`oneflow.atan`
    """
)

add_docstr(
    oneflow.Tensor.arctan, 
    """
    arctan() -> Tensor
    See :func:`oneflow.arctan`
    """
)

add_docstr(
    oneflow.Tensor.ceil, 
    """
    ceil() -> Tensor
    See :func:`oneflow.ceil`
    """
)

add_docstr(
    oneflow.Tensor.clamp, 
    """
    clamp() -> Tensor
    See :func:`oneflow.clamp`
    """
)

add_docstr(
    oneflow.Tensor.clip, 
    """
    clip() -> Tensor
    See :func:`oneflow.clip`
    """
)

add_docstr(
    oneflow.Tensor.cos, 
    """
    cos() -> Tensor
    See :func:`oneflow.cos`
    """
)

add_docstr(
    oneflow.Tensor.cosh, 
    """
    cosh() -> Tensor
    See :func:`oneflow.cosh`
    """
)

add_docstr(
    oneflow.Tensor.erf, 
    """
    erf() -> Tensor
    See :func:`oneflow.erf`
    """
)

add_docstr(
    oneflow.Tensor.erfc, 
    """
    erfc() -> Tensor
    See :func:`oneflow.erfc`
    """
)

add_docstr(
    oneflow.Tensor.expm1, 
    """
    expm1() -> Tensor
    See :func:`oneflow.expm1`
    """
)

add_docstr(
    oneflow.Tensor.fmod, 
    """
    fmod() -> Tensor
    See :func:`oneflow.fmod`
    """
)

add_docstr(
    oneflow.Tensor.log, 
    """
    log() -> Tensor
    See :func:`oneflow.log`
    """
)

add_docstr(
    oneflow.Tensor.minimum, 
    """
    minimum() -> Tensor
    See :func:`oneflow.minimum`
    """
)

add_docstr(
    oneflow.Tensor.maximum, 
    """
    maximum() -> Tensor
    See :func:`oneflow.maximum`
    """
)

add_docstr(
    oneflow.Tensor.pow, 
    """
    pow() -> Tensor
    See :func:`oneflow.pow`
    """
)

add_docstr(
    oneflow.Tensor.rsqrt, 
    """
    rsqrt() -> Tensor
    See :func:`oneflow.rsqrt`
    """
)

add_docstr(
    oneflow.Tensor.sqrt, 
    """
    sqrt() -> Tensor
    See :func:`oneflow.sqrt`
    """
)

add_docstr(
    oneflow.Tensor.square, 
    """
    square() -> Tensor
    See :func:`oneflow.square`
    """
)

add_docstr(
    oneflow.Tensor.matmul, 
    """
    matmul() -> Tensor
    See :func:`oneflow.matmul`
    """
)

add_docstr(
    oneflow.Tensor.round, 
    """
    round() -> Tensor
    See :func:`oneflow.round`
    """
)

add_docstr(
    oneflow.Tensor.softplus, 
    """
    softplus() -> Tensor
    See :func:`oneflow.softplus`
    """
)

add_docstr(
    oneflow.Tensor.tril, 
    """
    tril() -> Tensor
    See :func:`oneflow.tril`
    """
)

add_docstr(
    oneflow.Tensor.triu, 
    """
    triu() -> Tensor
    See :func:`oneflow.triu`
    """
=======
    oneflow.tensor,
    r"""
    Constructs a tensor with data, return a consistent tensor if placement and sbp are in kwargs,
       otherwise return a local tensor. 
       
    Arguments:
        data: Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar or tensor.
    Keyword Arguments:
        dtype (oneflow.dtype, optional) – the desired data type of returned tensor.
            Default: if None, infers data type from data.
        device (oneflow.device, optional): the desired device of returned tensor. If placement
            and sbp is None, uses the current cpu for the default tensor type.
        placement (oneflow.placement, optional): the desired placement of returned tensor.
        sbp (oneflow.sbp or tuple of oneflow.sbp, optional): the desired sbp of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False

    Noted:
        The Keyword Argument device is mutually exclusive with placement and sbp.
        Consistent tensor only can be constructed from tensor.


    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([1,2,3])
        >>> x
        tensor([1, 2, 3], dtype=oneflow.int64)

    """,
>>>>>>> origin/dev_tensor_functional_api
)
