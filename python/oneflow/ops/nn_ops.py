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
import collections

def calc_pool_padding(padding, dhw_offset, ndims):
    if isinstance(padding, str):
        padding = "SAME_LOWER" if padding.upper() == "SAME" else padding
        assert padding.upper() in ["VALID", "SAME_LOWER", "SAME_UPPER"]
        padding_type = padding.lower()
        ndim_pads_list = [[0, 0]] * ndims
    elif isinstance(padding, (list, tuple)):
        padding_type = "customized"
        ndim_pads_list = get_ndim_pads_list(padding, dhw_offset, ndims)
    else:
        raise ValueError("padding must be str or a list.")
    return (padding_type, ndim_pads_list)


def _GetSequence(value, n, name):
    """Formats value from input"""
    if value is None:
        value = [1]
    elif not isinstance(value, collections.Sized):
        value = [value]
    current_n = len(value)
    if current_n == 1:
        return list(value * n)
    elif current_n == n:
        return list(value)
    else:
        raise ValueError(
            "{} should be of length 1 or {} but was {}".format(name, n, current_n)
        )


def get_dhw_offset(channel_pos):
    if channel_pos == "channels_first":
        return 2
    else:
        return 1


def get_ndim_pads_list(padding, dhw_offset, ndims):
    pads_list = []
    for i in range(len(padding)):
        pad = padding[i]
        if isinstance(pad, int):
            pad = [pad, pad]
        elif isinstance(pad, (list, tuple)):
            assert len(pad) == 2
            pad = [pad[0], pad[1]]
        else:
            raise ValueError("padding must be list tuple or int")
        if i in range(dhw_offset, dhw_offset + ndims):
            pads_list.append(pad)
        else:
            assert pad == [0, 0]
    return pads_list
