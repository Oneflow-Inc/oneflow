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
import traceback
import warnings
from contextlib import contextmanager

import oneflow._oneflow_internal


def split_sbp(dim=None, **kwargs) -> oneflow._oneflow_internal.sbp.sbp:
    """
    Generate a split signature which indicates the tensor will be split along `dim`.

    Args:
        dim (int): The dimension in which the tensor is split. 

    Returns:
        SbpParallel: Split scheme object, often required by `to_global` method of `Tensor`

    Example::
        array = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        t1 = flow.tensor(array)
        ct2 = t1.to_global(sbp=flow.sbp.split(0), placement=("cuda", ranks=[0, 1, 2, 3]))

    """
    if dim is None:
        for key, value in kwargs.items():
            if key == "axis":
                if not isinstance(value, int):
                    raise TypeError(
                        "split_sbp(): parameter must be int, not {}.".format(
                            type(value)
                        )
                    )
                warnings.warn(
                    "This 'axis' parameter of oneflow.sbp.split() has been updated to 'dim' since OneFlow version 0.8."
                )
                dim = value
            else:
                raise TypeError(
                    "split_sbp() got an unexpected keyword argument '%s'." % key
                )

        if dim is None:
            raise TypeError("split_sbp() missing 1 required argument: 'dim'.")

    else:
        for key, value in kwargs.items():
            if key == "axis":
                raise TypeError(
                    "split_sbp() received an invalid combination of arguments - duplicate argument `axis`"
                )
            else:
                raise TypeError(
                    "split_sbp() got an unexpected keyword argument '%s'." % key
                )

    assert isinstance(dim, int)
    return oneflow._oneflow_internal.sbp.split(dim)
