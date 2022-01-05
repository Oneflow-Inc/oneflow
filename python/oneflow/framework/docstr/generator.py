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

oneflow.Generator.__doc__ = r"""
    flow.Generator(device='auto') → Generator

    Creates and returns a generator object that manages the state of the algorithm which produces pseudo random numbers. 
    Used as a keyword argument in many In-place random sampling functions.
       
    Keyword Arguments:
        device (str, optional) – the desired device for the generator.
            Default: "auto".

    Note:
        When device is "auto", the auto-generator will automatically infer the actual device (cpu/cuda). 
        For example, when a auto generator is used for cpu op, a cpu-generator will be created and stored.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> g_auto = flow.Generator()
        >>> g_auto = flow.Generator(device='auto')

    """


oneflow.Generator.device.__doc__ = r"""
    Generator.device -> device

    Gets the current device of the generator.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> g_auto = flow.Generator()
        >>> g_auto.device
        device(type='auto', index=0)

    """

oneflow.Generator.manual_seed.__doc__ = r"""
    Sets the seed for generating random numbers. Returns a torch.Generator object. 
    It is recommended to set a large seed, i.e. a number that has a good balance of 0 and 1 bits. 
    Avoid having many 0 bits in the seed.
       
    Parameter:
        seed (int) – The desired seed. 
        Value must be within the inclusive range [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]. 
        Otherwise, a RuntimeError is raised. 
        Negative inputs are remapped to positive values with the formula 0xffff_ffff_ffff_ffff + seed.


    For example:

    .. code-block:: python

        >>> g_auto.manual_seed(2147483647)
        <oneflow._oneflow_internal.Generator object at 0x7f3a1cfba1a8>
        >>> g_auto
        <oneflow._oneflow_internal.Generator object at 0x7f3a1cfba1a8>

    """


add_docstr(
    oneflow.Generator.set_state.__func__,
    r"""
    Sets the Generator state.

    For example:

    .. code-block:: python

        >>> g_auto = flow.Generator()
        >>> g_auto_other = flow.Generator()
        >>> g_auto.set_state(g_auto_other.get_state())

    """,
)

add_docstr(
    oneflow.Generator.get_state.__func__,
    r"""
    Returns the Generator state as a flow.Tensor.

    For example:

    .. code-block:: python

        >>> g_auto = flow.Generator()
        >>> g_auto.get_state()
        tensor([  1, 209, 156, 241,  48,  61,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], dtype=oneflow.uint8)

    """,
)

add_docstr(
    oneflow.Generator.seed.__func__,
    r"""
    Gets a non-deterministic random number from std::random_device or the current time and uses it to seed a Generator.

    For example:

    .. code-block:: python

        >>> g_auto = flow.Generator()
        >>> g_auto.seed()
        8267152500171235

    """,
)

add_docstr(
    oneflow.Generator.initial_seed.__func__,
    r"""
    Returns the initial seed for generating random numbers.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> g_auto.initial_seed()
        67280421310721

    """,
)
