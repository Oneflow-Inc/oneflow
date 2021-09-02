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
    oneflow.Tensor.abs, 
    r"""
    abs() -> Tensor
    See :func:`oneflow.abs`
    """
)

add_docstr(
    oneflow.Tensor.exp, 
    r"""
    exp() -> Tensor
    See :func:`oneflow.exp`
    """
)

add_docstr(
    oneflow.Tensor.acos, 
    r"""
    acos() -> Tensor
    See :func:`oneflow.acos`
    """
)

add_docstr(
    oneflow.Tensor.acosh, 
    r"""
    acosh() -> Tensor
    See :func:`oneflow.acosh`
    """
)

add_docstr(
    oneflow.Tensor.arccosh, 
    r"""
    arccosh() -> Tensor
    See :func:`oneflow.arccosh`
    """
)

add_docstr(
    oneflow.Tensor.atanh, 
    r"""
    atanh() -> Tensor
    See :func:`oneflow.atanh`
    """
)

add_docstr(
    oneflow.Tensor.arctanh, 
    r"""
    arctanh() -> Tensor
    See :func:`oneflow.arctanh`
    """
)

add_docstr(
    oneflow.Tensor.sign, 
    r"""
    sign() -> Tensor
    See :func:`oneflow.sign`
    """
)

add_docstr(
    oneflow.Tensor.sinh, 
    r"""
    sinh() -> Tensor
    See :func:`oneflow.sinh`
    """
)

add_docstr(
    oneflow.Tensor.tan, 
    r"""
    tan() -> Tensor
    See :func:`oneflow.tan`
    """
)

add_docstr(
    oneflow.Tensor.gt, 
    r"""
    gt() -> Tensor
    See :func:`oneflow.gt`
    """
)

add_docstr(
    oneflow.Tensor.ge, 
    r"""
    ge() -> Tensor
    See :func:`oneflow.ge`
    """
)

add_docstr(
    oneflow.Tensor.gelu, 
    r"""
    gelu() -> Tensor
    See :func:`oneflow.gelu`
    """
)

add_docstr(
    oneflow.Tensor.mish, 
    r"""
    mish() -> Tensor
    See :func:`oneflow.mish`
    """
)

add_docstr(
    oneflow.Tensor.sigmoid, 
    r"""
    sigmoid() -> Tensor
    See :func:`oneflow.sigmoid`
    """
)

add_docstr(
    oneflow.Tensor.tanh, 
    r"""
    tanh() -> Tensor
    See :func:`oneflow.tanh`
    """
)

add_docstr(
    oneflow.Tensor.silu, 
    r"""
    silu() -> Tensor
    See :func:`oneflow.silu`
    """
)

add_docstr(
    oneflow.Tensor.selu, 
    r"""
    selu() -> Tensor
    See :func:`oneflow.selu`
    """
)

add_docstr(
    oneflow.Tensor.softsign, 
    r"""
    softsign() -> Tensor
    See :func:`oneflow.softsign`
    """
)

add_docstr(
    oneflow.Tensor.cast, 
    r"""
    cast() -> Tensor
    See :func:`oneflow.cast`
    """
)

add_docstr(
    oneflow.Tensor.log1p, 
    r"""
    log1p() -> Tensor
    See :func:`oneflow.log1p`
    """
)

add_docstr(
    oneflow.Tensor.add, 
    r"""
    add() -> Tensor
    See :func:`oneflow.add`
    """
)

add_docstr(
    oneflow.Tensor.add_, 
    r"""
    add_() -> Tensor
    See :func:`oneflow.add_`
    """
)

add_docstr(
    oneflow.Tensor.div, 
    r"""
    div() -> Tensor
    See :func:`oneflow.div`
    """
)

add_docstr(
    oneflow.Tensor.mul, 
    r"""
    mul() -> Tensor
    See :func:`oneflow.mul`
    """
)


add_docstr(
    oneflow.Tensor.reciprocal, 
    r"""
    reciprocal() -> Tensor
    See :func:`oneflow.reciprocal`
    """
)

add_docstr(
    oneflow.Tensor.sub, 
    r"""
    sub() -> Tensor
    See :func:`oneflow.sub`
    """
)

add_docstr(
    oneflow.Tensor.asin, 
    r"""
    asin() -> Tensor
    See :func:`oneflow.asin`
    """
)

add_docstr(
    oneflow.Tensor.arcsin, 
    r"""
    arcsin() -> Tensor
    See :func:`oneflow.arcsin`
    """
)

add_docstr(
    oneflow.Tensor.asinh, 
    r"""
    asinh() -> Tensor
    See :func:`oneflow.asinh`
    """
)

add_docstr(
    oneflow.Tensor.arcsinh, 
    r"""
    arcsinh() -> Tensor
    See :func:`oneflow.arcsinh`
    """
)

add_docstr(
    oneflow.Tensor.atan, 
    r"""
    atan() -> Tensor
    See :func:`oneflow.atan`
    """
)

add_docstr(
    oneflow.Tensor.arctan, 
    r"""
    arctan() -> Tensor
    See :func:`oneflow.arctan`
    """
)

add_docstr(
    oneflow.Tensor.ceil, 
    r"""
    ceil() -> Tensor
    See :func:`oneflow.ceil`
    """
)

add_docstr(
    oneflow.Tensor.clamp, 
    r"""
    clamp() -> Tensor
    See :func:`oneflow.clamp`
    """
)

add_docstr(
    oneflow.Tensor.clip, 
    r"""
    clip() -> Tensor
    See :func:`oneflow.clip`
    """
)

add_docstr(
    oneflow.Tensor.cos, 
    r"""
    cos() -> Tensor
    See :func:`oneflow.cos`
    """
)

add_docstr(
    oneflow.Tensor.cosh, 
    r"""
    cosh() -> Tensor
    See :func:`oneflow.cosh`
    """
)

add_docstr(
    oneflow.Tensor.erf, 
    r"""
    erf() -> Tensor
    See :func:`oneflow.erf`
    """
)

add_docstr(
    oneflow.Tensor.erfc, 
    r"""
    erfc() -> Tensor
    See :func:`oneflow.erfc`
    """
)

add_docstr(
    oneflow.Tensor.expm1, 
    r"""
    expm1() -> Tensor
    See :func:`oneflow.expm1`
    """
)

add_docstr(
    oneflow.Tensor.fmod, 
    r"""
    fmod() -> Tensor
    See :func:`oneflow.fmod`
    """
)

add_docstr(
    oneflow.Tensor.log, 
    r"""
    log() -> Tensor
    See :func:`oneflow.log`
    """
)

add_docstr(
    oneflow.Tensor.minimum, 
    r"""
    minimum() -> Tensor
    See :func:`oneflow.minimum`
    """
)

add_docstr(
    oneflow.Tensor.maximum, 
    r"""
    maximum() -> Tensor
    See :func:`oneflow.maximum`
    """
)

add_docstr(
    oneflow.Tensor.pow, 
    r"""
    pow() -> Tensor
    See :func:`oneflow.pow`
    """
)

add_docstr(
    oneflow.Tensor.rsqrt, 
    r"""
    rsqrt() -> Tensor
    See :func:`oneflow.rsqrt`
    """
)

add_docstr(
    oneflow.Tensor.sqrt, 
    r"""
    sqrt() -> Tensor
    See :func:`oneflow.sqrt`
    """
)

add_docstr(
    oneflow.Tensor.square, 
    r"""
    square() -> Tensor
    See :func:`oneflow.square`
    """
)

add_docstr(
    oneflow.Tensor.matmul, 
    r"""
    matmul() -> Tensor
    See :func:`oneflow.matmul`
    """
)

add_docstr(
    oneflow.Tensor.round, 
    r"""
    round() -> Tensor
    See :func:`oneflow.round`
    """
)

add_docstr(
    oneflow.Tensor.softplus, 
    r"""
    softplus() -> Tensor
    See :func:`oneflow.softplus`
    """
)

add_docstr(
    oneflow.Tensor.tril, 
    r"""
    tril() -> Tensor
    See :func:`oneflow.tril`
    """
)

add_docstr(
    oneflow.Tensor.triu, 
    r"""
    triu() -> Tensor
    See :func:`oneflow.triu`
    """
)
