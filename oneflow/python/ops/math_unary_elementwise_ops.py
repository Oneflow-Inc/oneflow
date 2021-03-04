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
from __future__ import absolute_import

import os
import traceback
from typing import Optional

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.ops.user_op_builder as user_op_builder
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api


def build_unary_elemwise_math_op(math_op, x, name=None):
    if name is None:
        name = id_util.UniqueStr(math_op + "_")
    return (
        flow.user_op_builder(name)
        .Op(math_op)
        .Input("x", [x])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.abs")
def abs(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator returns the absolute value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def abs_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.abs(x)


        x = np.array([-1, 2, -3]).astype(np.float32)
        out = abs_Job(x)

        # out [1. 2. 3.]

    """
    return build_unary_elemwise_math_op("abs", x, name)


@oneflow_export("math.acos")
def acos(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the acos value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def acos_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.acos(x)


        x = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        out = acos_Job(x)

        # out [1.0471976 0.9272952 0.7953989]
        # We take the first value as an example 
        # (arccos(0.5) * pi) / 180 = 1.0471976
        
    """
    return build_unary_elemwise_math_op("acos", x, name)


@oneflow_export("math.acosh")
def acosh(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the inverse hyperbolic cosine value of Blob.

    The equation is: 

    .. math:: 
    
        out = log(x+(x^2-1)^\frac{1}{2})
    
    Args:
        x (oneflow_api.BlobDesc): A Blob, the range is [1, inf]
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def acosh_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.acosh(x)


        x = np.array([2, 3, 4]).astype(np.float32)
        out = acosh_Job(x)

        # out [1.316958  1.7627473 2.063437 ]

    """
    return build_unary_elemwise_math_op("acosh", x, name)


@oneflow_export("math.asin")
def asin(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the arcsin value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def asin_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.asin(x)


        x = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        out = asin_Job(x)

        # out [0.5235988  0.64350116 0.7753975 ]
        # We take the first value as an example 
        # (arcsin(0.5) * pi) / 180 = 0.5235988

    """
    return build_unary_elemwise_math_op("asin", x, name)


@oneflow_export("math.asinh")
def asinh(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the inverse hyperbolic sine value of Blob.

    The equation is: 

    .. math:: 
    
        out = log(x+(x^2+1)^\frac{1}{2})
    
    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def asinh_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.asinh(x)


        x = np.array([2, 3, 4]).astype(np.float32)
        out = asinh_Job(x)

        # out [1.4436355 1.8184464 2.0947125]
        
    """
    return build_unary_elemwise_math_op("asinh", x, name)


@oneflow_export("math.atan")
def atan(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the arctan value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def atan_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.atan(x)


        x = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        out = atan_Job(x)

        # out [0.4636476  0.5404195  0.61072594]
        # We take the first value as an example 
        # (arctan(0.5) * pi) / 180 = 0.4636476

    """
    return build_unary_elemwise_math_op("atan", x, name)


@oneflow_export("math.atanh")
def atanh(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the inverse hyperbolic tangent value of Blob.

    The equation is: 

    .. math:: 

        out = \frac{1}{2}*log(\frac{1+x}{1-x})

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 
    
    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def atanh_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.atanh(x)


        x = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        out = atanh_Job(x)

        # out [0.54930615 0.6931472  0.8673005 ]

    """
    return build_unary_elemwise_math_op("atanh", x, name)


@oneflow_export("math.ceil")
def ceil(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the ceiling value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def ceil_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.ceil(x)


        x = np.array([1.3, 1.5, 2.7]).astype(np.float32)
        out = ceil_Job(x)

        # out [2. 2. 3.]

    """
    return build_unary_elemwise_math_op("ceil", x, name)


@oneflow_export("math.cos")
def cos(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the cosine value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def cos_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.cos(x)


        x = np.array([1/3*np.pi, 0.25*np.pi, 1.25*np.pi]).astype(np.float32)
        out = cos_Job(x)

        # out [ 0.49999997  0.70710677 -0.7071068 ]

    """
    return build_unary_elemwise_math_op("cos", x, name)


@oneflow_export("math.cosh")
def cosh(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes hyperbolic cosine value of Blob.

    The equation is: 

    .. math:: 

        out = \frac{e^x+e^{-x}}{2}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def cosh_Job(x: tp.Numpy.Placeholder((3,))
                    ) -> tp.Numpy:
            return flow.math.cosh(x)


        x = np.array([1, 2, 3]).astype(np.float32)
        out = cosh_Job(x)

        # out [ 1.5430806  3.7621958 10.067662 ]

    """
    return build_unary_elemwise_math_op("cosh", x, name)


@oneflow_export("math.erf")
def erf(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the Gauss error value of Blob.

    The equation is: 

    .. math ::
    
        out = \frac{2}{\sqrt{\pi}}*\int_{0}^{x}e^{-z^2}\mathrm{d}{z}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def erf_Job(x: tp.Numpy.Placeholder((3,))
                    ) -> tp.Numpy:
            return flow.math.erf(x)


        x = np.array([1, 2, 3]).astype(np.float32)
        out = erf_Job(x)

        # out [0.8427008 0.9953223 0.9999779]

    """
    return build_unary_elemwise_math_op("erf", x, name)


@oneflow_export("math.erfc")
def erfc(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the :math:`1-erf(x)`, for more details of `erf` function 
    please refer to `math.erf`.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def erfc_Job(x: tp.Numpy.Placeholder((3,))
                    ) -> tp.Numpy:
            return flow.math.erfc(x)


        x = np.array([1, 2, 3]).astype(np.float32)
        out = erfc_Job(x)

        # out [1.5729921e-01 4.6777353e-03 2.2090495e-05]

    """
    return build_unary_elemwise_math_op("erfc", x, name)


@oneflow_export("math.exp")
def exp(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the exponential of Blob.

    The equation is: 

    .. math:: 

        out = e^x

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def exp_Job(x: tp.Numpy.Placeholder((3,))
                    ) -> tp.Numpy:
            return flow.math.exp(x)


        x = np.array([1, 2, 3]).astype(np.float32)
        out = exp_Job(x)

        # out [ 2.7182817  7.389056  20.085537 ]

    """
    return build_unary_elemwise_math_op("exp", x, name)


@oneflow_export("math.expm1")
def expm1(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes :math:`y=e^x-1`.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def expm1_Job(x: tp.Numpy.Placeholder((3,))
                    ) -> tp.Numpy:
            return flow.math.expm1(x)


        x = np.array([1, 2, 3]).astype(np.float32)
        out = expm1_Job(x)

        # out [ 1.7182819  6.389056  19.085537 ]

    """
    return build_unary_elemwise_math_op("expm1", x, name)


@oneflow_export("math.floor")
def floor(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the largest integer not greater than input Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def floor_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.floor(x)


        x = np.array([1.3, 1.5, 2.7]).astype(np.float32)
        out = floor_Job(x)

        # out [1. 1. 2.]

    """
    return build_unary_elemwise_math_op("floor", x, name)


@oneflow_export("math.lgamma")
def lgamma(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the :math:`Gamma(x)` value.

    The equation is: 

    .. math:: 

        out = \int_{0}^{\infty}t^{x-1}*e^{-t}\mathrm{d}{t}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def lgamma_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.lgamma(x)


        x = np.array([1.3, 1.5, 2.7]).astype(np.float32)
        out = lgamma_Job(x)

        # out [-0.1081748  -0.12078223  0.4348206 ]

    """
    return build_unary_elemwise_math_op("lgamma", x, name)


@oneflow_export("math.log")
def log(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the log value of input Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def log_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.log(x)


        x = np.array([1.3, 1.5, 2.7]).astype(np.float32)
        out = log_Job(x)

        # out [0.26236424 0.40546513 0.9932518 ]

    """
    return build_unary_elemwise_math_op("log", x, name)


@oneflow_export("math.log1p")
def log1p(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the :math:`log(x)+1` value of input Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def log1p_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.log1p(x)


        x = np.array([1.3, 1.5, 2.7]).astype(np.float32)
        out = log1p_Job(x)

        # out [0.8329091  0.91629076 1.3083328 ]

    """
    return build_unary_elemwise_math_op("log1p", x, name)


@oneflow_export("math.log_sigmoid")
def log_sigmoid(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""This operator computes the log sigmoid value of input Blob.

    The equation is: 

    .. math:: 

        out = log(\frac{1}{1+e^{-x}})

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def log_sigmoid_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.log_sigmoid(x)


        x = np.array([1.3, 1.5, 2.7]).astype(np.float32)
        out = log_sigmoid_Job(x)

        # out [-0.24100842 -0.20141333 -0.0650436 ]

    """
    return build_unary_elemwise_math_op("log_sigmoid", x, name)


@oneflow_export("math.negative")
def negative(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """This operator computes the negative value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def negative_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.negative(x)


        x = np.array([1.3, 1.5, 2.7]).astype(np.float32)
        out = negative_Job(x)

        # out [-1.3 -1.5 -2.7]

    """
    return build_unary_elemwise_math_op("negative", x, name)


@oneflow_export("math.reciprocal")
def reciprocal(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""This operator computes the reciprocal of x.

    The equation is: 

    .. math:: 

        out = \frac{1}{x}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reciprocal_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.reciprocal(x)


        x = np.array([1, 2, 4]).astype(np.float32)
        out = reciprocal_Job(x)

        # out [1.   0.5  0.25]

    """
    return build_unary_elemwise_math_op("reciprocal", x, name)


@oneflow_export("math.reciprocal_no_nan")
def reciprocal_no_nan(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """This operator computes the safe reciprocal of x. If x is zero, the reciprocal will 
    be also set to zero.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reciprocal_no_nan_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.reciprocal_no_nan(x)


        x = np.array([0, 2, 4]).astype(np.float32)
        out = reciprocal_no_nan_Job(x)

        # out [0.   0.5  0.25]

    """
    return build_unary_elemwise_math_op("reciprocal_no_nan", x, name)


@oneflow_export("math.rint")
def rint(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the closest integer to Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def rint_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.rint(x)


        x = np.array([1.49999, 1.500001, 2.7]).astype(np.float32)
        out = rint_Job(x)

        # out [1. 2. 3.]

    """
    return build_unary_elemwise_math_op("rint", x, name)


@oneflow_export("math.round")
def round(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator rounds the value of Blob to the nearest integer. 

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 
    
    .. code-block:: python 
    
        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def round_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.round(x)


        x = np.array([1.49999, 1.500001, 2.7]).astype(np.float32)
        out = round_Job(x)

        # out [1. 2. 3.]

    """
    return build_unary_elemwise_math_op("round", x, name)


@oneflow_export("math.rsqrt")
def rsqrt(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the reciprocal of square root value of Blob.

    The equation is: 

    .. math:: 

        out=\frac{1}{\sqrt{x}}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def rsqrt_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.rsqrt(x)


        x = np.array([4, 16, 25]).astype(np.float32)
        out = rsqrt_Job(x)

        # out [0.5  0.25 0.2 ]

    """
    return build_unary_elemwise_math_op("rsqrt", x, name)


@oneflow_export("math.sigmoid_v2")
def sigmoid_v2(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""This operator computes the sigmoid value of Blob. 

    The equation is: 

    .. math:: 

        out=\frac{1}{1+e^{-x}}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sigmoidv2_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.sigmoid_v2(x)

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        out = sigmoidv2_Job(x)

        # out [0.37754068 0.5        0.62245935]

    """
    return build_unary_elemwise_math_op("sigmoid_v2", x, name)


@oneflow_export("math.sign")
def sign(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator returns the sign of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sign_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.sign(x)


        x = np.array([-2, 0, 2]).astype(np.float32)
        out = sign_Job(x)

        # out [-1.  0.  1.]

    """
    return build_unary_elemwise_math_op("sign", x, name)


@oneflow_export("math.sin")
def sin(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the sin value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sin_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.sin(x)


        x = np.array([-1/6*np.pi, 0, 1/6*np.pi]).astype(np.float32)
        out = sin_Job(x)

        # out [-0.5  0.   0.5]

    """
    return build_unary_elemwise_math_op("sin", x, name)


@oneflow_export("math.sinh")
def sinh(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the hyperbolic sine value of Blob.

    The equation is: 

    .. math:: 

        out =\frac{e^x-e^{-x}}{2}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sinh_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.sinh(x)


        x = np.array([-1, 0, 1]).astype(np.float32)
        out = sinh_Job(x)

        # out [-1.1752012  0.         1.1752012]

    """
    return build_unary_elemwise_math_op("sinh", x, name)


@oneflow_export("math.softplus")
def softplus(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """This operator computes the softplus value of Blob.

    The equation is: 

    .. math:: 

        out = log(e^x+1)

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def softplus_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.softplus(x)


        x = np.array([-1, 0, 1]).astype(np.float32)
        out = softplus_Job(x)

        # out [0.31326166 0.6931472  1.3132616 ]

    """
    return build_unary_elemwise_math_op("softplus", x, name)


@oneflow_export("math.sqrt")
def sqrt(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the sqrt root value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sqrt_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.sqrt(x)


        x = np.array([4, 16, 25]).astype(np.float32)
        out = sqrt_Job(x)

        # out [2. 4. 5.]

    """
    return build_unary_elemwise_math_op("sqrt", x, name)


@oneflow_export("math.square")
def square(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the square value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def square_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.square(x)


        x = np.array([2, 3, 4]).astype(np.float32)
        out = square_Job(x)

        # out [ 4.  9. 16.]

    """
    return build_unary_elemwise_math_op("square", x, name)


@oneflow_export("math.tan")
def tan(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    """This operator computes the tan value of Blob.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def tan_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.tan(x)


        x = np.array([-1/4*np.pi, 0, 1/4*np.pi]).astype(np.float32)
        out = tan_Job(x)

        # out [-1.  0.  1.]

    """
    return build_unary_elemwise_math_op("tan", x, name)


@oneflow_export("math.tanh")
def tanh(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""This operator computes the hyperbolic tangent value of Blob.

    The equation is: 

    .. math:: 

        out = \frac{e^x-e^{-x}}{e^x+e^{-x}}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def tanh_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.tanh(x)


        x = np.array([-1, 0, 1]).astype(np.float32)
        out = tanh_Job(x)
    
        # out [-0.7615942  0.         0.7615942]

    """
    return build_unary_elemwise_math_op("tanh", x, name)


@oneflow_export("math.tanh_v2")
def tanh_v2(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""This operator computes the hyperbolic tangent value of Blob.

    The equation is:

    .. math::

        out = \frac{e^x-e^{-x}}{e^x+e^{-x}}

    Args:
        x (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob
    """

    print(
        """WARNING: flow.math.tanh_v2 has been deprecated. Please replace it by flow.math.tanh.
        """
    )
    print(traceback.format_stack()[-2])
    return flow.math.tanh(x, name)
