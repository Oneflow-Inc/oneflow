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
from oneflow._oneflow_internal.oneflow.core.common import data_type as data_type_cfg
from oneflow._oneflow_internal.oneflow.core.common import shape as shape_cfg
from oneflow._oneflow_internal.oneflow.core.framework import (
    user_op_attr as user_op_attr_cfg,
)
from oneflow.compatible import single_client as flow


def SetAttrValue(attr_value, py_value, default_attr_value):
    if default_attr_value.HasField("at_bool"):
        if py_value is None:
            py_value = True
        assert type(py_value) is bool
        attr_value.set_at_bool(py_value)
    elif default_attr_value.HasField("at_int64"):
        assert type(py_value) is int
        attr_value.set_at_int64(py_value)
    elif default_attr_value.HasField("at_double"):
        assert type(py_value) is float
        attr_value.set_at_double(py_value)
    elif default_attr_value.HasField("at_string"):
        assert type(py_value) is str
        attr_value.set_at_string(py_value)
    else:
        raise ValueError(
            "config with type %s is invalid. supported types: [bool, int, float, str]"
            % type(py_value)
        )


def convert_to_user_attr_value(op_type_name, attr_name, attr_value):
    attribute = user_op_attr_cfg.AttrValue()
    assert isinstance(attr_name, str)
    attr_type = oneflow._oneflow_internal.GetUserOpAttrType(op_type_name, attr_name)
    if attr_type == user_op_attr_cfg.kAtInt32:
        assert isinstance(attr_value, int)
        attribute.set_at_int32(attr_value)
    elif attr_type == user_op_attr_cfg.kAtInt64:
        assert isinstance(attr_value, int)
        attribute.set_at_int64(attr_value)
    elif attr_type == user_op_attr_cfg.kAtBool:
        assert isinstance(attr_value, bool)
        attribute.set_at_bool(attr_value)
    elif attr_type == user_op_attr_cfg.kAtFloat:
        assert isinstance(attr_value, (float, int))
        attribute.set_at_float(attr_value)
    elif attr_type == user_op_attr_cfg.kAtDouble:
        assert isinstance(attr_value, (float, int))
        attribute.set_at_double(attr_value)
    elif attr_type == user_op_attr_cfg.kAtString:
        assert isinstance(attr_value, str)
        attribute.set_at_string(attr_value)
    elif attr_type == user_op_attr_cfg.kAtShape:
        assert isinstance(attr_value, (tuple, list))
        attribute_mutable_at_shape = attribute.mutable_at_shape()
        for x in attr_value:
            assert isinstance(x, int)
            attribute_mutable_at_shape.add_dim(x)
    elif attr_type == user_op_attr_cfg.kAtDataType:
        assert attr_value in flow.dtypes()
        attr_value = oneflow._oneflow_internal.deprecated.GetProtoDtype4OfDtype(
            attr_value
        )
        assert isinstance(attr_value, int)
        attribute.set_at_data_type(data_type_cfg.DataType(attr_value))
    elif attr_type == user_op_attr_cfg.kAtListInt32:
        assert isinstance(attr_value, (tuple, list))
        attribute_mutable_at_list_int32 = attribute.mutable_at_list_int32()
        for x in attr_value:
            assert isinstance(x, int)
            attribute_mutable_at_list_int32.add_val(x)
    elif attr_type == user_op_attr_cfg.kAtListInt64:
        assert isinstance(attr_value, (tuple, list))
        attribute_mutable_at_list_int64 = attribute.mutable_at_list_int64()
        for x in attr_value:
            assert isinstance(x, int)
            attribute_mutable_at_list_int64.add_val(x)
    elif attr_type == user_op_attr_cfg.kAtListFloat:
        assert isinstance(attr_value, (tuple, list))
        attribute_mutable_at_list_float = attribute.mutable_at_list_float()
        for x in attr_value:
            assert isinstance(x, (float, int))
            attribute_mutable_at_list_float.add_val(x)
    elif attr_type == user_op_attr_cfg.kAtListDataType:
        assert isinstance(attr_value, (tuple, list))
        attribute_mutable_at_list_data_type = attribute.mutable_at_list_data_type()
        for x in attr_value:
            assert x in flow.dtypes()
            x = oneflow._oneflow_internal.deprecated.GetProtoDtype4OfDtype(x)
            assert isinstance(x, int)
            attribute_mutable_at_list_data_type.add_val(data_type_cfg.DataType(x))
    elif attr_type == user_op_attr_cfg.kAtListShape:
        assert isinstance(attr_value, (tuple, list))
        attribute_mutable_at_list_shape = (
            attribute.mutable_at_list_shape().mutable_val()
        )
        for x in attr_value:
            assert isinstance(x, (tuple, list))
            shape = shape_cfg.ShapeProto()
            for dim in x:
                assert isinstance(dim, int)
                shape.add_dim(dim)
            attribute_mutable_at_list_shape.Add().CopyFrom(shape)
    elif attr_type == user_op_attr_cfg.kAtListString:
        assert isinstance(attr_value, (tuple, list))
        attribute_mutable_at_list_string = attribute.mutable_at_list_string()
        for x in attr_value:
            assert isinstance(x, str)
            attribute_mutable_at_list_string.add_val(x)
    else:
        raise ValueError("Invalid op attribute type {}".format(attr_type))
    return attribute
