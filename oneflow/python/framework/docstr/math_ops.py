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
from oneflow.python.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.F.add,
    r"""
add(x, y) -> Tensor

Computes the addition of x by y for each element, the shape of 
tensor x and y should be same.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()
        >>> x = flow.Tensor(np.random.randn(2,3))
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

""",
)

add_docstr(
    oneflow.F.mul,
    r"""
mul(x, y) -> Tensor

Computes the multiplication of x by y for each element, the shape of 
tensor x and y should be same.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()
        >>> x = flow.Tensor(np.random.randn(2,3))
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.mul(x,y).numpy()
        >>> out.shape
        (2, 3)

""",
)
