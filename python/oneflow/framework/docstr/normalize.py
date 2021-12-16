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
    oneflow.nn.functional.normalize,
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html#torch.nn.functional.normalize
    
    Performs :math:`L_p` normalization of inputs over specified dimension.
    For a tensor input of sizes :math:`(n_0 , ...,n_{dim},...,n_k)` each :math:`n_{dim}`-element vector vv along
    dimension dim is transformed as :

    .. math::	
	​   v = \frac{v}{max(||v||_p, \epsilon )}

    With the default arguments it uses the Euclidean norm over vectors along dimension 11 for normalization.

    Args:
        input (Tensor): input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12 

    Returns:
        Tensor: the output Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.randn(3,4,5)
        >>> output = flow.nn.functional.normalize(input, 1, 2)
        >>> output.shape
        oneflow.Size([3, 4, 5])  

    """,
)
