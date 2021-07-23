import os
from collections import OrderedDict
import numpy as np
from oneflow.compatible import single_client as flow


class MemoryZoneOutOfMemoryException(Exception):
    def __init__(self, err="memory_zone_out_of_memory"):
        Exception.__init__(self, err)


def constant(device_type):
    flow.env.init()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    @flow.global_function(function_config=func_config)
    def ConstantJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.constant(
                6, dtype=flow.float, shape=(1024 * 1024 * 1024, 1024 * 1024 * 1024)
            )
            return x

    try:
        ConstantJob().get()
    except Exception as e:
        if "memory_zone_out_of_memory" in str(e):
            print(e)
            raise MemoryZoneOutOfMemoryException()


def memory_zone_out_of_memory_of_gpu():
    return constant("gpu")


def memory_zone_out_of_memory_of_cpu():
    return constant("cpu")
