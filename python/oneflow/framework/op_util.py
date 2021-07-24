"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow._oneflow_internal
from oneflow.core.operator.op_conf_pb2 import OperatorConf


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
