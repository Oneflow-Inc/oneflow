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
from oneflow.nn.module import Module


class Sequential(get_seq(Module)):
    """A sequential container.
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
    pass


class ModuleDict(get_dict(Module)):
    pass


class ParameterList(get_para_list(Module)):
    pass


class ParameterDict(get_para_dict(Module)):
    pass


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
