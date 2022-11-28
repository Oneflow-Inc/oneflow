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
    oneflow.set_num_threads,
    """
    Sets the number of threads used for intraop parallelism on CPU.
    
    .. WARNING::
        To ensure that the correct number of threads is used, 
        set_num_threads must be called before running eager, eager globe or ddp.

    """,
)

add_docstr(
    oneflow.get_default_dtype,
    """oneflow.get_default_dtype() -> oneflow._oneflow_internal.dtype

    Returns the default floating point dtype.

    Returns:
        oneflow.dtype: The default floating point dtype.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.set_default_dtype(flow.float32)
        >>> flow.get_default_dtype()
        oneflow.float32
        >>> flow.set_default_dtype(flow.float64)
        >>> flow.get_default_dtype()
        oneflow.float64
        >>> flow.set_default_tensor_type(flow.FloatTensor)
        >>> flow.get_default_dtype()
        oneflow.float32
    """,
)

add_docstr(
    oneflow.set_default_dtype,
    """oneflow.set_default_dtype() -> None

    Sets the default floating point type for those source operators which create Tensor.

    The default floating point type is ``oneflow.float32``.

    Args:
        dtype (oneflow.dtype): The floating point dtype.

    For example:

    .. code-block:: python

        >>> import oneflow
        >>> oneflow.set_default_dtype(oneflow.float64)
        >>> x = oneflow.randn(2, 3)
        >>> x.dtype
        oneflow.float64
        >>> oneflow.set_default_dtype(oneflow.float32)
        >>> x = oneflow.randn(2, 3)
        >>> x.dtype
        oneflow.float32
    """,
)
