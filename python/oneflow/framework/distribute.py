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


def split_sbp(*dim, **karg) -> oneflow._oneflow_internal.sbp.sbp:
    """Generate a split scheme in which op will be splitted at `dim`.

    Args:
        dim (int): At `dim` the op will be splitted.

    Returns:
        SbpParallel: Split scheme object, often required by `to_global` method of `Tensor`

    Example::
        array = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        t1 = flow.tensor(array)
        ct2 = t1.to_global(sbp=flow.sbp.split(0), placement=("cuda", ranks=[0, 1, 2, 3]))

    """
    if len(karg) != 0:
        try:
            dim = karg["dim"]
        except:
            dim = karg["axis"]
            warnings.warn(
                "This 'axis' parameter of oneflow.sbp.split() will be updated to 'dim' in version 0.8."
            )
    else:
        dim = dim[0]

    assert type(dim) is int
    return oneflow._oneflow_internal.sbp.split(dim)
