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
from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util


def DeviceType4DeviceTag(device_tag):
    global _device_tag2device_type
    if device_tag not in _device_tag2device_type:
        _device_tag2device_type[device_tag] = c_api_util.DeviceType4DeviceTag(
            device_tag
        )
    return _device_tag2device_type[device_tag]


_device_tag2device_type = {}
