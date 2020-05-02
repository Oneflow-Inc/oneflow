from __future__ import absolute_import

import oneflow.python.framework.g_func_ctx as g_func_ctx
from oneflow.core.operator.op_conf_pb2 import OperatorConf

def IsOpConfOnlyCpuSupported(op_conf):
    assert isinstance(op_conf, OperatorConf)
    """
    global _cpu_only_op_type_cases
    if _cpu_only_op_type_cases == None:
        _cpu_only_op_type_cases = set()
        for field in OperatorConf.DESCRIPTOR.oneofs_by_name["op_type"].fields:
            if g_func_ctx.IsOpTypeCaseCpuSupportOnly(field.number):
                _cpu_only_op_type_cases.add(field.number)
    op_type_field = op_conf.WhichOneof("op_type")
    return OperatorConf.DESCRIPTOR.fields_by_name[op_type_field].number in _cpu_only_op_type_cases
    """
    op_type_field = op_conf.WhichOneof("op_type")
    field_number = OperatorConf.DESCRIPTOR.fields_by_name[op_type_field].number
    return g_func_ctx.IsOpTypeCaseCpuSupportOnly(field_number)
    
# _cpu_only_op_type_cases = None
