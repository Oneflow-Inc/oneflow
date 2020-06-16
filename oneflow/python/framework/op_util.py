from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
from oneflow.core.operator.op_conf_pb2 import OperatorConf


def IsOpConfOnlyCpuSupported(op_conf):
    assert isinstance(op_conf, OperatorConf)
    """
    global _cpu_only_op_type_cases
    if _cpu_only_op_type_cases == None:
        _cpu_only_op_type_cases = set()
        for field in OperatorConf.DESCRIPTOR.oneofs_by_name["op_type"].fields:
            if c_api_util.IsOpTypeCaseCpuSupportOnly(field.number):
                _cpu_only_op_type_cases.add(field.number)
    op_type_field = op_conf.WhichOneof("op_type")
    return OperatorConf.DESCRIPTOR.fields_by_name[op_type_field].number in _cpu_only_op_type_cases
    """
    op_type_field = op_conf.WhichOneof("op_type")
    if op_type_field == "user_conf":
        return IsUserOpOnlyCpuSupported(op_conf.user_conf.op_type_name)
    else:
        field_number = OperatorConf.DESCRIPTOR.fields_by_name[op_type_field].number
        return c_api_util.IsOpTypeCaseCpuSupportOnly(field_number)


def IsUserOpOnlyCpuSupported(op_type_name):
    return c_api_util.IsOpTypeNameCpuSupportOnly(op_type_name)


# _cpu_only_op_type_cases = None
