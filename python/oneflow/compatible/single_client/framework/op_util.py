from oneflow.compatible.single_client.core.operator.op_conf_pb2 import OperatorConf
import oneflow._oneflow_internal


def IsOpConfOnlyCpuSupported(op_conf):
    assert isinstance(op_conf, OperatorConf)
    '\n    global _cpu_only_op_type_cases\n    if _cpu_only_op_type_cases == None:\n        _cpu_only_op_type_cases = set()\n        for field in OperatorConf.DESCRIPTOR.oneofs_by_name["op_type"].fields:\n            if oneflow._oneflow_internal.IsOpTypeCaseCpuSupportOnly(field.number):\n                _cpu_only_op_type_cases.add(field.number)\n    op_type_field = op_conf.WhichOneof("op_type")\n    return OperatorConf.DESCRIPTOR.fields_by_name[op_type_field].number in _cpu_only_op_type_cases\n    '
    op_type_field = op_conf.WhichOneof("op_type")
    if op_type_field == "user_conf":
        return IsUserOpOnlyCpuSupported(op_conf.user_conf.op_type_name)
    else:
        field_number = OperatorConf.DESCRIPTOR.fields_by_name[op_type_field].number
        return oneflow._oneflow_internal.IsOpTypeCaseCpuSupportOnly(field_number)


def IsUserOpOnlyCpuSupported(op_type_name):
    return oneflow._oneflow_internal.IsOpTypeNameCpuSupportOnly(op_type_name)
