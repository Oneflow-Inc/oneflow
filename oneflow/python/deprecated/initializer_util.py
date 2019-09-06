from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.common.data_type_pb2 as data_type_conf_util
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("truncated_normal")
def truncated_normal_initializer(stddev=1.0):
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.truncated_normal_conf, "std", float(stddev))

    return initializer
