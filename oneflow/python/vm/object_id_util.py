from __future__ import absolute_import

import oneflow.python.framework.hob as hob
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util

@oneflow_export("vm.new_physical_object_id", enable_if=hob.in_normal_mode & hob.env_initialized)
def NewPhysicalObjectId():
    return c_api_util.NewPhysicalObjectId()

@oneflow_export("vm.new_physical_symbol_id", enable_if=hob.in_normal_mode & hob.env_initialized)
def NewPhysicalSymbolId():
    return c_api_util.NewPhysicalSymbolId()
