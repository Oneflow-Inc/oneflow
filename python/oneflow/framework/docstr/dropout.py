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
    oneflow._C.dropout,
    """
    dropout(x: Tensor, p: float = 0.5, training: bool = True, generator :Generator = None, *, addend: Tensor) -> Tensor 
    
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.dropout.html.

    Args:      
        x(Tensor): A Tensor which will be applyed dropout. 
        p(float): probability of an element to be zeroed. Default: 0.5    
        training(bool): If is True it will apply dropout. Default: True     
        generator(Generator, optional):  A pseudorandom number generator for sampling
        addend(Tensor, optional):  A Tensor add in result after dropout, it can be used in model's residual connection structure. Default: None  

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example:

    Example 1: 

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

       
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> y = flow.nn.functional.dropout(x, p=0) 

        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> generator = flow.Generator()
        >>> y = flow.nn.functional.dropout(x, p=0.5, generator=generator) 
      
    Example 2: 
    
    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

       
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> addend = flow.ones((3, 4), dtype=flow.float32)
        >>> y = flow.nn.functional.dropout(x, p=0, addend=addend) 
        >>> y #doctest: +ELLIPSIS
        tensor([[ 0.2203,  1.2264,  1.2458,  1.4163],
                [ 1.4299,  1.3626,  0.5108,  1.4141],
                [-0.4115,  2.2183,  0.4497,  1.6520]], dtype=oneflow.float32)
    
    See :class:`~oneflow.nn.Dropout` for details.   
 
    """,
)

add_docstr(
    oneflow._C.dropout1d,
    r"""
    dropout1d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor 

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.dropout1d.html.

    Randomly zero out entire channels (a channel is a 1D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~oneflow.nn.Dropout1d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
    """,
)

add_docstr(
    oneflow._C.dropout2d,
    r"""
    dropout1d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor 

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.dropout2d.html.

    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~oneflow.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
    """,
)

add_docstr(
    oneflow._C.dropout3d,
    r"""
    dropout1d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor 

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.dropout3d.html.

    Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~oneflow.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
    """,
)

add_docstr(
    oneflow.nn.Dropout,
    """
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.Dropout.html.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    "Improving neural networks by preventing co-adaptation of feature
    detectors".

    Furthermore, the outputs are scaled by a factor of :math:`\\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Additionally, we can pass an extra Tensor `addend` which shape is consistent with input Tensor. 
    The `addend` Tensor will be add in result after dropout, it is very useful in model's residual connection structure.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        generator:  A pseudorandom number generator for sampling

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example:

    example 1: 

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.Dropout(p=0)
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[-0.7797,  0.2264,  0.2458,  0.4163],
                [ 0.4299,  0.3626, -0.4892,  0.4141],
                [-1.4115,  1.2183, -0.5503,  0.6520]], dtype=oneflow.float32)
    
    example 2: 
    
    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.Dropout(p=0)
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> addend = flow.ones((3, 4), dtype=flow.float32)
        >>> y = m(x, addend=addend)
        >>> y #doctest: +ELLIPSIS
        tensor([[ 0.2203,  1.2264,  1.2458,  1.4163],
                [ 1.4299,  1.3626,  0.5108,  1.4141],
                [-0.4115,  2.2183,  0.4497,  1.6520]], dtype=oneflow.float32)
    
    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """,
)

add_docstr(
    oneflow.nn.Dropout1d,
    """
    Randomly zero out entire channels (a channel is a 1D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.Dropout1d.html.

    Usually the input comes from :class:`nn.Conv1d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`oneflow.nn.Dropout1d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`.
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input).

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.Dropout1d(p=0)
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[-0.7797,  0.2264,  0.2458,  0.4163],
                [ 0.4299,  0.3626, -0.4892,  0.4141],
                [-1.4115,  1.2183, -0.5503,  0.6520]], dtype=oneflow.float32)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """,
)

add_docstr(
    oneflow.nn.Dropout2d,
    """
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.Dropout2d.html.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`oneflow.nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input).

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.Dropout2d(p=0)
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[-0.7797,  0.2264,  0.2458,  0.4163],
                [ 0.4299,  0.3626, -0.4892,  0.4141],
                [-1.4115,  1.2183, -0.5503,  0.6520]], dtype=oneflow.float32)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """,
)

add_docstr(
    oneflow.nn.Dropout3d,
    """
    Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.Dropout2d.html.

    Usually the input comes from :class:`nn.Conv3d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`oneflow.nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input).

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.Dropout3d(p=0)
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[-0.7797,  0.2264,  0.2458,  0.4163],
                [ 0.4299,  0.3626, -0.4892,  0.4141],
                [-1.4115,  1.2183, -0.5503,  0.6520]], dtype=oneflow.float32)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """,
)
