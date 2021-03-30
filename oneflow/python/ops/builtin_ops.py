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
from __future__ import absolute_import


import oneflow
import oneflow_api
import oneflow_api.oneflow.core.common.shape as shape_cfg
import oneflow_api.oneflow.core.common.data_type as data_type_cfg
import oneflow_api.oneflow.core.framework.user_op_attr as user_op_attr_cfg
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("builtin_op")
class BuiltinOp(object):
    def __init__(self, op_type_name, op_name=None):
        if op_name is None:
            self._builder = oneflow_api.one.OpBuilder(op_type_name)
        else:
            self._builder = oneflow_api.one.OpBuilder(op_type_name, op_name)
        self._op = None
        self._op_type_name = op_type_name

    @property
    def op(self):
        r"""access the builtin op

        Returns:
            the builtin op
        """
        # TODO: Check for op completeness.
        if self._op is None:
            self._op = self._builder.build()
        return self._op

    def Input(self, input_name, num=1):
        r"""Set input blob of op

        Args:
            input_name (str): input name of blob
            num (int, optional) : Defaults to 1.

        Returns:
            self
        """
        assert isinstance(num, int) and num >= 1
        self._builder.input(input_name, num)
        return self

    def Output(self, output_name, num=1):
        r"""Set output blob of op

        Args:
            output_name (str): name of output blob
            num (int, optional):  Defaults to 1.

        Returns:
            self
        """
        assert isinstance(num, int) and num >= 1
        self._builder.output(output_name, num)
        return self

    def Attr(self, attr_name, attr_value, attr_type_name=None):
        r"""Set value of op's attribute.

        Args:
            attr_name (str): attribute name of op
            attr_value (Any): attribute value of op

        Raises:
            ValueError: raised when value is not idential to op's attribute type.

        Returns:
            [type]: [description]
        """
        if attr_type_name is not None:
            print(
                """WARNING: Argument 'attr_type_name' of UserOpConfBuilder.Attr has been deprecated. Please remove it.

            For instance:
                -     .Attr("out_num", out_num, "AttrTypeInt64")
                +     .Attr("out_num", out_num)
                        """
            )
            print(traceback.format_stack()[-2])

        attribute = user_op_attr_cfg.AttrValue()
        assert isinstance(attr_name, str)
        assert self._op_type_name is not None
        attr_type = oneflow_api.GetUserOpAttrType(self._op_type_name, attr_name)
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
            assert attr_value in oneflow.dtypes()
            attr_value = oneflow_api.deprecated.GetProtoDtype4OfDtype(attr_value)
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
                assert x in oneflow.dtypes()
                x = oneflow_api.deprecated.GetProtoDtype4OfDtype(x)
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

        self._builder.attr(attr_name, attribute)
        return self

    def Build(self):
        r"""Explicitly complete the construction of the builtin op
        
        Returns:
            the completed builtin op
        """
        if self._op is None:
            self._op = self._builder.build()
        return self._op
