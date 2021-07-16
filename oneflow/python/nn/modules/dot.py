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
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api


@oneflow_export("dot")
@experimental_api
def dot(input, other):
    r"""This operator computes the dot product of tensor input and other.
    
    The equation is:
    
	$$		
	​   \\sum_{i=1}^{n}(x[i] * y[i])
	$$
    
    Args:
        input (Tensor):  first tensor in the dot product.
        other (Tensor):  second tensor in the dot product.

    Shape:
        - input: Input must be 1D.
        - other: Other must be 1D.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()
        >>> flow.dot(flow.Tensor([2, 3]), flow.Tensor([2, 1]))
        tensor(7)
        
    """
    return flow.F.dot(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
