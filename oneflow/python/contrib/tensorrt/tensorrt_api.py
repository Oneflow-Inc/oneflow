from __future__ import absolute_import

import oneflow.oneflow_internal as oneflow_internal
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('tensorrt.write_int8_calibration')
def write_int8_calibration(path):
    oneflow_internal.WriteInt8Calibration(path)
