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
from oneflow.nn.utils.container import *
from oneflow.nn.modules.module import Module


class Sequential(get_seq(Module)):
    """A sequential container.

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.Sequential.html?#torch.nn.Sequential.
    
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example:

    .. code-block:: python

        >>> import oneflow.nn as nn
        >>> from collections import OrderedDict
        >>> nn.Sequential(nn.Conv2d(1,20,5), nn.ReLU(), nn.Conv2d(20,64,5), nn.ReLU()) #doctest: +ELLIPSIS
        Sequential(
          (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
          (1): ReLU()
          (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
          (3): ReLU()
        )
        >>> nn.Sequential(OrderedDict([
        ...    ('conv1', nn.Conv2d(1,20,5)),
        ...    ('relu1', nn.ReLU()),
        ...    ('conv2', nn.Conv2d(20,64,5)),
        ...    ('relu2', nn.ReLU())
        ... ])) #doctest: +ELLIPSIS
        Sequential(
          (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
          (relu1): ReLU()
          (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
          (relu2): ReLU()
        )

    """

    pass


class ModuleList(get_list(Module)):
    """Holds submodules in a list.

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.ModuleList.html?#torch.nn.ModuleList.
    
    :class:`~oneflow.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~oneflow.nn.Module` methods.
    
    Args:
        modules (iterable, optional): an iterable of modules to add
    
    .. code-block:: python

        >>> import oneflow.nn as nn

        >>> class MyModule(nn.Module):
        ...    def __init__(self):
        ...        super(MyModule, self).__init__()
        ...        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
        ...    def forward(self, x):
        ...        # ModuleList can act as an iterable, or be indexed using ints
        ...        for i, l in enumerate(self.linears):
        ...            x = self.linears[i // 2](x) + l(x)
        ...        return x

        >>> model = MyModule()
        >>> model.linears
        ModuleList(
          (0): Linear(in_features=10, out_features=10, bias=True)
          (1): Linear(in_features=10, out_features=10, bias=True)
          (2): Linear(in_features=10, out_features=10, bias=True)
          (3): Linear(in_features=10, out_features=10, bias=True)
          (4): Linear(in_features=10, out_features=10, bias=True)
          (5): Linear(in_features=10, out_features=10, bias=True)
          (6): Linear(in_features=10, out_features=10, bias=True)
          (7): Linear(in_features=10, out_features=10, bias=True)
          (8): Linear(in_features=10, out_features=10, bias=True)
          (9): Linear(in_features=10, out_features=10, bias=True)
        )
        

    """

    pass


class ModuleDict(get_dict(Module)):
    """Holds submodules in a dictionary.

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.ModuleDict.html?#torch.nn.ModuleDict.

    :class:`~oneflow.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~oneflow.nn.Module` methods.

    :class:`~oneflow.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~oneflow.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~oneflow.nn.ModuleDict` (the argument to
      :meth:`~oneflow.nn.ModuleDict.update`).

    Note that :meth:`~oneflow.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    .. code-block:: python

        >>> import oneflow.nn as nn

        >>> class MyModule(nn.Module):
        ...    def __init__(self):
        ...        super(MyModule, self).__init__()
        ...        self.choices = nn.ModuleDict({
        ...                'conv': nn.Conv2d(10, 10, 3),
        ...                'pool': nn.MaxPool2d(3)
        ...        })
        ...        self.activations = nn.ModuleDict([
        ...                ['lrelu', nn.LeakyReLU()],
        ...                ['prelu', nn.PReLU()]
        ...        ])

        ...    def forward(self, x, choice, act):
        ...        x = self.choices[choice](x)
        ...        x = self.activations[act](x)
        ...        return x
    
        >>> model = MyModule()
        >>> model.choices
        ModuleDict(
          (conv): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
          (pool): MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=(0, 0), dilation=(1, 1))
        )
    """

    pass


class ParameterList(get_para_list(Module)):
    """Holds parameters in a list.

    :class:`~oneflow.nn.ParameterList` can be indexed like a regular Python
    list, but parameters it contains are properly registered, and will be
    visible by all :class:`~oneflow.nn.Module` methods.

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.ParameterList.html?#torch.nn.ParameterList.

    Args:
        parameters (iterable, optional): an iterable of :class:`~oneflow.nn.Parameter` to add

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> class MyModule(nn.Module):
        ...    def __init__(self):
        ...        super(MyModule, self).__init__()
        ...        self.params = nn.ParameterList([nn.Parameter(flow.randn(10, 10)) for i in range(10)])
        ...
        ...    def forward(self, x):
        ...        # ParameterList can act as an iterable, or be indexed using ints
        ...        for i, p in enumerate(self.params):
        ...            x = self.params[i // 2].mm(x) + p.mm(x)
        ...        return x

        >>> model = MyModule()
        >>> model.params
        ParameterList(
            (0): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (1): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (2): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (3): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (4): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (5): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (6): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (7): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (8): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
            (9): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 10x10]
        )
    """

    pass


class ParameterDict(get_para_dict(Module)):
    """
    Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~oneflow.nn.ParameterDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~oneflow.nn.ParameterDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~oneflow.nn.ParameterDict` (the argument to
      :meth:`~oneflow.nn.ParameterDict.update`).

    Note that :meth:`~oneflow.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.
    
    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.ParameterDict.html?#torch.nn.ParameterDict.

    Args:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~oneflow.nn.Parameter`) or an iterable of key-value pairs
            of type (string, :class:`~oneflow.nn.Parameter`)

    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> class MyModule(nn.Module):
        ...    def __init__(self):
        ...        super(MyModule, self).__init__()
        ...        self.params = nn.ParameterDict({
        ...                'left': nn.Parameter(flow.randn(5, 10)),
        ...                'right': nn.Parameter(flow.randn(5, 10))
        ...        })
        ...
        ...    def forward(self, x, choice):
        ...        x = self.params[choice].mm(x)
        ...        return x

        >>> model = MyModule()
        >>> model.params
        ParameterDict(
            (left): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 5x10]
            (right): Parameter containing: [<class 'oneflow.nn.Parameter'> of size 5x10]
        )
    """

    pass


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
