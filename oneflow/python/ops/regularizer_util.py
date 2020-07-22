from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("regularizers.l1_l2")
def l1_l2_regularizer(
    l1: float = 0.01, l2: float = 0.01
) -> op_conf_util.RegularizerConf:
    regularizer = op_conf_util.RegularizerConf()
    setattr(regularizer.l1_l2_conf, "l1", l1)
    setattr(regularizer.l1_l2_conf, "l2", l2)
    return regularizer


@oneflow_export("regularizers.l1")
def l1_regularizer(l: float = 0.01) -> op_conf_util.RegularizerConf:
    return l1_l2_regularizer(l1=l, l2=0.0)


@oneflow_export("regularizers.l2")
def l2_regularizer(l: float = 0.01) -> op_conf_util.RegularizerConf:
    return l1_l2_regularizer(l1=0.0, l2=l)
