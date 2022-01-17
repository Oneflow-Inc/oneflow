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
r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
import re
import collections

import oneflow as flow


string_classes = (str, bytes)

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, (flow.Tensor, flow._oneflow_internal.Tensor)):
        return data
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        # array of string classes and object
        if (
            elem_type.__name__ == "ndarray"
            and np_str_obj_array_pattern.search(data.dtype.str) is not None
        ):
            return data
        return flow.tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, collections.abc.Sequence) and not isinstance(
        data, string_classes
    ):
        return [default_convert(d) for d in data]
    else:
        # NOTE: pytorch just return data here, and not raise any exception!
        raise TypeError(default_convert_err_msg_format.format(elem_type))


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

default_convert_err_msg_format = (
    "default_convert: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, (flow.Tensor, flow._oneflow_internal.Tensor)):
        # TODO: tensor.storage()._new_shared(numel)
        return flow._C.stack(batch, dim=0)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([flow.tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return flow.tensor(batch)
    elif isinstance(elem, float):
        return flow.tensor(batch, dtype=flow.float64)
    elif isinstance(elem, int):
        return flow.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
