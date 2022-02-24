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

oneflow.placement.__doc__ = r"""
    A oneflow.palcement is an object representing the device group on which a oneflow.Tensor is or will be allocated. The oneflow.palcement contains a device type ('cpu' or 'cuda') and corresponding device sequence.
    
    A oneflow.Tensor's palcement can be accessed via the Tensor.palcement property.
    
    A oneflow.palcement can be constructed in several ways:
    
    .. code-block:: python

        >>> import oneflow as flow
        
        >>> p = flow.placement("cuda", ranks=[0, 1, 2, 3])
        >>> p
        oneflow.placement(type="cuda", ranks=[0, 1, 2, 3])
        >>> p = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        >>> p
        oneflow.placement(type="cuda", ranks=[[0, 1], [2, 3]])
        
    """
