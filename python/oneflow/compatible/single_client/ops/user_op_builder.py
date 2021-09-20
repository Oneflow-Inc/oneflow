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
import random
import traceback

from google.protobuf import text_format

import oneflow._oneflow_internal
from oneflow._oneflow_internal.oneflow.core.common import data_type as data_type_cfg
from oneflow._oneflow_internal.oneflow.core.common import shape as shape_cfg
from oneflow._oneflow_internal.oneflow.core.framework import (
    user_op_attr as user_op_attr_cfg,
)
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.eager import eager_blob_util as eager_blob_util
from oneflow.compatible.single_client.eager import gradient_util as gradient_util
from oneflow.compatible.single_client.experimental import namescope as name_scope
from oneflow.compatible.single_client.framework import c_api_util as c_api_util
from oneflow.compatible.single_client.framework import (
    compile_context as compile_context,
)
from oneflow.compatible.single_client.framework import distribute as distribute
from oneflow.compatible.single_client.framework import hob as hob
from oneflow.compatible.single_client.framework import interpret_util as interpret_util
from oneflow.compatible.single_client.framework import remote_blob as remote_blob_util
from oneflow.compatible.single_client.support import enable_if as enable_if
from oneflow.core.eager import eager_symbol_pb2 as eager_symbol_util
from oneflow.core.framework import user_op_attr_pb2 as attr_value_pb
from oneflow.core.operator import op_conf_pb2 as op_conf_util
from oneflow.core.register import logical_blob_id_pb2 as logical_blob_id_util

blob_register = oneflow._oneflow_internal.GetDefaultBlobRegister()


class UserOp(object):
    def __init__(self, op_name, op_type_name=None):
        self.op_conf_ = op_conf_util.OperatorConf()
        self.op_conf_.name = op_name
        if op_type_name is not None:
            self.op_conf_.user_conf.op_type_name = op_type_name
        device_tag = flow.current_scope().device_parallel_desc_symbol.device_tag
        self.op_conf_.device_tag = device_tag
        self.output_arg_key_list_ = []

    @property
    def op_conf(self):
        return self.op_conf_

    def InferAndTryRun(self):
        raise NotImplementedError

    def MakeRemoteBlob(self, lbi):
        raise NotImplementedError

    def RemoteBlobList(self):
        remote_blob_list = []
        for k in self.op_conf_.user_conf.output:
            if k not in self.output_arg_key_list_:
                raise ValueError(
                    "output_arg_name {} of {} op is not set in python op builder".format(
                        k, self.op_conf_.name
                    )
                )
        for output_arg_name in self.output_arg_key_list_:
            assert output_arg_name in self.op_conf_.user_conf.output
            for i in range(len(self.op_conf_.user_conf.output[output_arg_name].s)):
                lbi = logical_blob_id_util.LogicalBlobId()
                lbi.op_name = self.op_conf_.name
                lbi.blob_name = "{}_{}".format(output_arg_name, i)
                remote_blob_obj = self.MakeRemoteBlob(lbi)
                remote_blob_list.append(remote_blob_obj)
                if flow.eager_execution_enabled():
                    gradient_util.GetDefaultBackwardBlobRegister().TrySetObject4BlobName(
                        remote_blob_obj.logical_blob_name, remote_blob_obj.blob_object
                    )
        return tuple(remote_blob_list)

    def RemoteBlobDict(self):
        remote_blob_dict = {}
        for k in self.op_conf_.user_conf.output:
            if k not in self.output_arg_key_list_:
                raise ValueError(
                    "output_arg_name {} of {} op is not set in python op builder".format(
                        k, self.op_conf_.name
                    )
                )
        for output_arg_name in self.output_arg_key_list_:
            assert output_arg_name in self.op_conf_.user_conf.output
            if output_arg_name not in remote_blob_dict:
                remote_blob_dict[output_arg_name] = []
            for i in range(len(self.op_conf_.user_conf.output[output_arg_name].s)):
                lbi = logical_blob_id_util.LogicalBlobId()
                lbi.op_name = self.op_conf_.name
                lbi.blob_name = "{}_{}".format(output_arg_name, i)
                remote_blob_dict[output_arg_name].append(self.MakeRemoteBlob(lbi))
        return remote_blob_dict

    def SoleOutputBlob(self):
        blobs = self.RemoteBlobList()
        assert len(blobs) == 1
        return blobs[0]


class UserOpModule(object):
    @property
    def opkernel_object(self):
        return self.opkernel_object_

    def set_opkernel_object(self, opkernel_object):
        assert not hasattr(self, "opkernel_object_")
        self.opkernel_object_ = opkernel_object

    def InitOpKernel(self):
        raise NotImplementedError


def api_user_op_builder(op_name):
    """Build a wrapper of user op.

    For instance::
        def myargmax(
            input: oneflow._oneflow_internal.BlobDesc) -> oneflow._oneflow_internal.BlobDesc:
            return (
            flow.user_op_builder("myargmax")
            .Op("argmax")
            .Input("in", [input])
            .Output("out")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
            )

    Args:
        op_name (str): name of new user op

    Returns:
        UserOpConfBuilder: `UserOpConfBuilder` object used to build a wrapper of user op.
    """
    api = enable_if.unique([lazy_user_op_builder, eager_user_op_builder])
    return api(op_name)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def lazy_user_op_builder(op_name):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_name
    return UserOpConfBuilder(LazyUserOp, op_name, None)


class LazyUserOp(UserOp):
    def __init__(self, op_name, op_type_name):
        UserOp.__init__(self, op_name, op_type_name)

    def InferAndTryRun(self):
        compile_context.CurJobAddOp(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.RemoteBlob(lbi)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def eager_user_op_builder(op_name):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_name
    return UserOpConfBuilder(EagerUserOp, op_name, None)


class EagerUserOp(UserOp):
    def __init__(self, op_name, op_type_name):
        UserOp.__init__(self, op_name, op_type_name)

    def InferAndTryRun(self):
        interpret_util.Forward(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.EagerLogicalBlob(lbi)


def api_consistent_user_op_builder(op_name):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_name
    return UserOpConfBuilder(ConsistentUserOp, op_name, None)


class ConsistentUserOp(UserOp):
    def __init__(self, op_name, op_type_name):
        UserOp.__init__(self, op_name, op_type_name)

    def InferAndTryRun(self):
        interpret_util.ConsistentForward(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.RemoteBlob(lbi)


class UserOpConfBuilder(object):
    def __init__(self, user_op_or_module_class, op_name, op_type_name):
        self.user_op_ = user_op_or_module_class(op_name, op_type_name)

    def CheckAndComplete(self):
        assert self.user_op_.op_conf_.user_conf.op_type_name != ""
        self.user_op_.op_conf_ = c_api_util.CheckAndCompleteUserOpConf(
            self.user_op_.op_conf_
        )
        return self

    def Build(self):
        """Build op when in/output and other attribute set up.

        Returns:
            self

        """
        return self.CheckAndComplete().user_op_

    def OpName(self, op_name):
        job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
        op_name = name_scope.GetJobNameScopePrefix(job_name) + op_name
        self.user_op_.op_conf_.name = op_name
        user_conf = self.user_op_.op_conf_.user_conf

        def GetLbn(output_name, i):
            return "{}/{}_{}".format(op_name, output_name, i)

        for (output_name, output) in user_conf.output.items():
            output.s[:] = [GetLbn(output_name, i) for i in range(len(output.s))]
        return self

    def Op(self, op_type_name):
        """set typename of op

        Args:
            op_type_name (string): op type name

        Returns:
            self
        """
        self.user_op_.op_conf_.user_conf.op_type_name = op_type_name
        return self

    def Input(self, input_name, input_blob_list):
        """Set input blob of op

        Args:
            input_name (str): input name of blob
            input_blob_list : list of blobs

        Returns:
            self
        """
        assert isinstance(input_blob_list, (tuple, list))
        input_conf = self.user_op_.op_conf_.user_conf.input
        if input_name not in input_conf:
            self.user_op_.op_conf_.user_conf.input_order.append(input_name)
        input_conf[input_name].ClearField("s")
        for input_blob in input_blob_list:
            input_conf[input_name].s.append(input_blob.unique_name)
        return self

    def InputSize(self, input_name, input_blob_size):
        input_conf = self.user_op_.op_conf_.user_conf.input
        assert input_blob_size >= 0
        assert input_name not in input_conf
        self.user_op_.op_conf_.user_conf.input_order.append(input_name)
        for i in range(input_blob_size):
            unique_name = "%s/%s_%s" % (self.user_op_.op_conf_.name, input_name, i)
            input_conf[input_name].s.append(unique_name)
        return self

    def Output(self, output_name, num=1):
        """Set output blob of op

        Args:
            output_name (str): name of output blob
            num (int, optional):  Defaults to 1.

        Returns:
            self
        """
        assert isinstance(num, int) and num >= 1
        out_lbns = []
        for i in range(num):
            lbn = "{}/{}_{}".format(self.user_op_.op_conf_.name, output_name, i)
            out_lbns.append(lbn)
        self.user_op_.op_conf_.user_conf.output[output_name].s[:] = out_lbns
        self.user_op_.op_conf_.user_conf.output_order.append(output_name)
        self.user_op_.output_arg_key_list_.append(output_name)
        return self

    def Attr(self, attr_name, attr_value, attr_type_name=None):
        """Set value of op's attribute.

        Args:
            attr_name (str): attribute name of op
            attr_value (Any): attribute value of op

        Raises:
            ValueError: raised when value is not idential to op's attribute type.

        Returns:
            [type]: [description]
        """
        if attr_type_name != None:
            print(
                'WARNING: Argument \'attr_type_name\' of UserOpConfBuilder.Attr has been deprecated. Please remove it.\n\n            For instance:\n                -     .Attr("out_num", out_num, "AttrTypeInt64")\n                +     .Attr("out_num", out_num)\n                        '
            )
            print(traceback.format_stack()[-2])
        attribute = user_op_attr_cfg.AttrValue()
        assert isinstance(attr_name, str)
        attr_type = oneflow._oneflow_internal.GetUserOpAttrType(
            self.user_op_.op_conf_.user_conf.op_type_name, attr_name
        )
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
        self.user_op_.op_conf_.user_conf.attr[attr_name].CopyFrom(
            text_format.Parse(str(attribute), attr_value_pb.AttrValue())
        )
        return self


def api_user_op_module_builder(op_type_name):
    api = enable_if.unique(
        [lazy_user_op_module_builder, eager_logical_user_op_module_builder]
    )
    return api(op_type_name)


class UserOpModuleBuilder(UserOpConfBuilder):
    def __init__(self, *args, **kwargs):
        UserOpConfBuilder.__init__(self, *args, **kwargs)
        self.user_op_module.op_conf.scope_symbol_id = flow.current_scope().symbol_id

    @property
    def user_op_module(self):
        return self.user_op_

    def Op(self, op_type_name):
        raise ValueError(
            "user op module builder of {} can't call '.Op(op_type_name)' method".format(
                op_type_name
            )
        )


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def lazy_user_op_module_builder(op_type_name):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_type_name
    return UserOpModuleBuilder(LazyUserOpModule, op_name, op_type_name)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def eager_logical_user_op_module_builder(op_type_name):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_type_name
    return UserOpModuleBuilder(EagerLogicalUserOpModule, op_name, op_type_name)


class LazyUserOpModule(UserOpModule, UserOp):
    def __init__(self, op_name, op_type_name):
        UserOp.__init__(self, op_name, op_type_name)

    def InitOpKernel(self):
        self.set_opkernel_object(None)

    def InferAndTryRun(self):
        assert hob.in_global_mode(None)
        compile_context.CurJobAddOp(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.RemoteBlob(lbi)


class EagerLogicalUserOpModule(UserOpModule, UserOp):
    def __init__(self, op_name, op_type_name):
        UserOp.__init__(self, op_name, op_type_name)

    def InitOpKernel(self):
        def BuildInstruction(builder):
            if not isinstance(
                self.op_conf,
                oneflow._oneflow_internal.oneflow.core.operator.op_conf.OperatorConf,
            ):
                cfg_op_conf = oneflow._oneflow_internal.deprecated.MakeOpConfByString(
                    str(self.op_conf)
                )
            self.set_opkernel_object(builder.NewOpKernelObject(cfg_op_conf))

        oneflow._oneflow_internal.deprecated.LogicalRun(BuildInstruction)

    def InferAndTryRun(self):
        assert hob.in_global_mode(None)
        interpret_util.OpKernelForward(self.op_conf, self.opkernel_object)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.EagerLogicalBlob(lbi)


def api_consistent_user_op_module_builder(op_type_name):
    api = enable_if.unique(
        [
            lazy_consistent_user_op_module_builder,
            eager_consistent_user_op_module_builder,
        ]
    )
    return api(op_type_name)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def lazy_consistent_user_op_module_builder(op_type_name):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_type_name
    return UserOpModuleBuilder(LazyConsistentUserOpModule, op_name, op_type_name)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def eager_consistent_user_op_module_builder(op_type_name):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_type_name
    return UserOpModuleBuilder(EagerConsistentUserOpModule, op_name, op_type_name)


class LazyConsistentUserOpModule(UserOpModule, UserOp):
    def __init__(self, op_name, op_type_name):
        UserOp.__init__(self, op_name, op_type_name)

    def InitOpKernel(self):
        self.set_opkernel_object(None)

    def InferAndTryRun(self):
        assert hob.in_global_mode(None)
        compile_context.CurJobAddConsistentOp(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.RemoteBlob(lbi)


class EagerConsistentUserOpModule(UserOpModule, UserOp):
    def __init__(self, op_name, op_type_name):
        UserOp.__init__(self, op_name, op_type_name)

    def InitOpKernel(self):
        def BuildInstruction(builder):
            if not isinstance(
                self.op_conf,
                oneflow._oneflow_internal.oneflow.core.operator.op_conf.OperatorConf,
            ):
                cfg_op_conf = oneflow._oneflow_internal.deprecated.MakeOpConfByString(
                    str(self.op_conf)
                )
            self.set_opkernel_object(builder.NewOpKernelObject(cfg_op_conf))

        oneflow._oneflow_internal.deprecated.LogicalRun(BuildInstruction)

    def InferAndTryRun(self):
        assert hob.in_global_mode(None)
        interpret_util.OpKernelConsistentForward(self.op_conf, self.opkernel_object)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.EagerLogicalBlob(lbi)
