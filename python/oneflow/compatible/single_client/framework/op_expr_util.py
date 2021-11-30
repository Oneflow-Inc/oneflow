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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.attr_util import (
    convert_to_user_attr_value,
)


def user_op_expr_call(self, *args, **kwargs):
    attrs = oneflow._oneflow_internal.MutableCfgAttrMap()
    for (attr_name, attr_value) in kwargs.items():
        assert isinstance(attr_name, str)
        attrs[attr_name] = convert_to_user_attr_value(
            self.op_type_name, attr_name, attr_value
        )
    try:
        results = self.apply(args, attrs)
    except oneflow._oneflow_internal.exception.Exception:
        raise oneflow._oneflow_internal.exception.GetThreadLocalLastError()
    return results


def RegisterMethod4UserOpExpr():
    oneflow._oneflow_internal.one.UserOpExpr.__call__ = user_op_expr_call
