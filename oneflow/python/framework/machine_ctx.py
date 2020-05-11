import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob

import oneflow.python.oneflow_export as oneflow_export

@oneflow_export('current_machine_id', enable_if=hob.env_initialized)
def CurrentMachineId():
  return c_api_util.CurrentMachineId()
