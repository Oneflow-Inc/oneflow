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
import sys
import traceback
import oneflow as flow
from collections import defaultdict
from typing import Any, DefaultDict, List, Optional


class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):
        return self


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = "Caught {} {}.\nOriginal {}".format(
            self.exc_type.__name__, self.where, self.exc_msg
        )
        if self.exc_type == KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            raise self.exc_type(message=msg)
        raise self.exc_type(msg)


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    The api is referenced from https://github.com/pytorch/pytorch/blob/master/torch/_utils.py#L437

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return flow._C.flatten(tensors[0])
    else:
        flatten_tensors = []
        for tensor in tensors:
            flatten_tensors.append(flow.flatten(tensor))
        return flow.cat(flatten_tensors, 0)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    The api is referenced from https://github.com/pytorch/pytorch/blob/master/torch/_utils.py#L474

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        if numel == 0:
            outputs.append(flow.zeros_like(tensor))
        else:
            outputs.append(flow.narrow(flat, 0, offset, numel).view(tensor.size()))
            offset += numel
    return outputs


def _take_tensors(tensors, size_limit):
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
    buf_dict: DefaultDict[str, List] = defaultdict(lambda: [[], 0])
    for tensor in tensors:
        t = tensor.type()
        size = tensor.numel() * tensor.element_size()
        buf_and_size = buf_dict[t]
        if buf_and_size[1] + size > size_limit and buf_and_size[1] > 0:
            yield buf_and_size[0]
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append(tensor)
        buf_and_size[1] += size
    for buf, _ in buf_dict.values():
        if len(buf) > 0:
            yield buf


def _reorder_tensors_as(tensors, ordered_tensors):
    """Assume that tensors are of same order as ordered_tensors within their
    types, e.g., from _take_tensors. Reorder them to be of same order as
    ordered_tensors.

    Args:
        tensors (Iterable[Tensor]): tensors to be reordered. They should be of
          the same order as ordered_tensors within their own types.
        ordered_tensors (Iterable[Tensor]): tensors whose order will be the
          reference.

    Returns:
        Ordered tuple of tensors with contents from tensors and order of
        ordered_tensors.
    """
    type_dict = defaultdict(list)
    for tensor in tensors:
        type_dict[tensor.type()].append(tensor)
    type_dict_ = {t: iter(coll) for t, coll in type_dict.items()}
    return tuple(next(type_dict_[tensor.type()]) for tensor in ordered_tensors)
