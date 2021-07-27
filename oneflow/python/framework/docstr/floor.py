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
    oneflow.F.floor
    r"""oneflow.F.floor
    The floor function takes a input x, and outpus the greates integer less than or equal to x, that is
    
    .. math::
        \lfoor{x}\rfloor = max{m \in \mathbb{Z} | m \le x}
    
    Args:
        x(tensor, dtype=flow.float32): the input real number
        output(tensor, dtype=flow.float32)
    
    For example:

    ..code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor(15.37)
        >>> x1 = flow.F.floor(x)
        >>> x1
        tensor([15.], dtype=oneflow.float32)
    
        
    """,
)
if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)