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

def relu_op(input,inplace=False):
    """
    Applies the rectified linear unit function element-wise. See :class:`~oneflow.nn.ReLU` for more details. oneflow space

    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    
    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> ndarr = np.asarray([1, -2, 3])
        >>> input = flow.Tensor(ndarr)
        >>> outputs = flow.relu(input)
        >>> outputs
        tensor([1., 0., 3.], dtype=oneflow.float32)

    """
    return flow.F.relu(input,inplace=inplace)

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
