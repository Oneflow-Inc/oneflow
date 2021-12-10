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

import oneflow._oneflow_internal
from oneflow._oneflow_internal.autograd import AutoGradMode


class no_grad(AutoGradMode):
    r"""
    Context-manager that disabled gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure that
    you will not call Tensor.backward(). It will reduce memory consumption for computations
    that would otherwise have requires_grad=True.

    In this mode, the result of every computation will have requires_grad=False, even when
    the inputs have requires_grad=True.

    This context manager is thread local; it will not affect computation in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    """

    def __init__(self):
        super().__init__(False)

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with AutoGradMode(False):
                return func(*args, **kwargs)

        return wrapper
