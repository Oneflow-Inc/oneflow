from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util

def DeviceType4DeviceTag(device_tag):
    global _device_tag2device_type
    if device_tag not in _device_tag2device_type:
        _device_tag2device_type[device_tag] = c_api_util.DeviceType4DeviceTag(device_tag)
    return _device_tag2device_type[device_tag]

_device_tag2device_type = {}
