from __future__ import absolute_import

import oneflow.python.framework.g_func_ctx as g_func_ctx

def DeviceType4DeviceTag(device_tag):
    global _device_tag2device_type
    if device_tag not in _device_tag2device_type:
        _device_tag2device_type[device_tag] = g_func_ctx.DeviceType4DeviceTag(device_tag)
    return _device_tag2device_type[device_tag]

_device_tag2device_type = {}
