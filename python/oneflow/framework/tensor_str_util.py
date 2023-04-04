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
import os
import oneflow as flow
from typing import Optional, Tuple


def _autoset_linewidth():
    # os.terminal_size(columns, lines),
    # columns represents width of the terminal window in characters
    # and lines represents height of the terminal window in characters.
    try:
        linewidth = os.get_terminal_size()[0]
    except OSError:
        linewidth = 80
    return linewidth


def _try_convert_to_local_tensor(tensor):
    if tensor.is_global:
        tensor = tensor.to_global(
            placement=flow.placement.all(tensor.placement.type), sbp=flow.sbp.broadcast,
        ).to_local()
    return tensor
