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
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.distribute as distribute
import oneflow.python.framework.hob as hob
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.framework.user_op_attr_pb2 as attr_value_pb
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.common.shape_pb2 as shape_util
import oneflow
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.hob as hob
import oneflow.python.experimental.name_scope as name_scope
import oneflow.core.eager.eager_symbol_pb2 as eager_symbol_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.eager_blob_util as eager_blob_util
import oneflow.python.lib.core.enable_if as enable_if
import random
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow as flow
import oneflow_api
import traceback

blob_register = blob_register_util.GetDefaultBlobRegister()


class UserOp(object):
    def __init__(self, op_name, op_type_name=None):
        self.op_conf_ = op_conf_util.OperatorConf()
        self.op_conf_.name = op_name
        if op_type_name is not None:
            self.op_conf_.user_conf.op_type_name = op_type_name
        device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
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


@oneflow_export("user_op_builder")
def api_user_op_builder(op_name):
    r"""Build a wrapper of user op.

    For instance::
        def myargmax(
            input: oneflow_api.BlobDesc) -> oneflow_api.BlobDesc:
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
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
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
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
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


@oneflow_export("consistent_user_op_builder")
def api_consistent_user_op_builder(op_name):
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
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
        r"""Build op when in/output and other attribute set up.

        Returns:
            self

        """
        return self.CheckAndComplete().user_op_

    def OpName(self, op_name):
        job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
        op_name = name_scope.GetJobNameScopePrefix(job_name) + op_name

        self.user_op_.op_conf_.name = op_name
        user_conf = self.user_op_.op_conf_.user_conf

        def GetLbn(output_name, i):
            return "{}/{}_{}".format(op_name, output_name, i)

        for output_name, output in user_conf.output.items():
            output.s[:] = [GetLbn(output_name, i) for i in range(len(output.s))]
        return self

    def Op(self, op_type_name):
        r"""set typename of op

        Args:
            op_type_name (string): op type name

        Returns:
            self
        """
        self.user_op_.op_conf_.user_conf.op_type_name = op_type_name
        return self

    def Input(self, input_name, input_blob_list):
        r"""Set input blob of op
       
        Args:
            input_name (str): input name of blob
            input_blob_list : list of blobs

        Returns:
            self
        """
        assert isinstance(input_blob_list, (tuple, list))
        input_conf = self.user_op_.op_conf_.user_conf.input
        input_conf[input_name].ClearField("s")
        for input_blob in input_blob_list:
            # assert type(input_blob) is blob_desc.BlobDesc
            input_conf[input_name].s.append(input_blob.unique_name)
        return self

    def InputSize(self, input_name, input_blob_size):
        input_conf = self.user_op_.op_conf_.user_conf.input
        assert input_blob_size >= 0
        assert input_name not in input_conf
        for i in range(input_blob_size):
            unique_name = "%s/%s_%s" % (self.user_op_.op_conf_.name, input_name, i)
            input_conf[input_name].s.append(unique_name)
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
        out_lbns = []
        for i in range(num):
            lbn = "{}/{}_{}".format(self.user_op_.op_conf_.name, output_name, i)
            out_lbns.append(lbn)
        self.user_op_.op_conf_.user_conf.output[output_name].s[:] = out_lbns
        self.user_op_.output_arg_key_list_.append(output_name)
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
        if attr_type_name != None:
            print(
                """WARNING: Argument 'attr_type_name' of UserOpConfBuilder.Attr has been deprecated. Please remove it.
            
            For instance:
                -     .Attr("out_num", out_num, "AttrTypeInt64")
                +     .Attr("out_num", out_num)
                        """
            )
            print(traceback.format_stack()[-2])

        attribute = attr_value_pb.AttrValue()
        assert isinstance(attr_name, str)
        attr_type = oneflow_api.GetUserOpAttrType(
            self.user_op_.op_conf_.user_conf.op_type_name, attr_name
        )
        if attr_type == attr_value_pb.kAtInt32:
            assert isinstance(attr_value, int)
            attribute.at_int32 = attr_value
        elif attr_type == attr_value_pb.kAtInt64:
            assert isinstance(attr_value, int)
            attribute.at_int64 = attr_value
        elif attr_type == attr_value_pb.kAtBool:
            assert isinstance(attr_value, bool)
            attribute.at_bool = attr_value
        elif attr_type == attr_value_pb.kAtFloat:
            assert isinstance(attr_value, float)
            attribute.at_float = attr_value
        elif attr_type == attr_value_pb.kAtDouble:
            assert isinstance(attr_value, float)
            attribute.at_double = attr_value
        elif attr_type == attr_value_pb.kAtString:
            assert isinstance(attr_value, str)
            attribute.at_string = attr_value
        elif attr_type == attr_value_pb.kAtShape:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_shape.dim[:] = list(attr_value)
        elif attr_type == attr_value_pb.kAtDataType:
            assert (
                isinstance(
                    oneflow_api.deprecated.GetProtoDtype4OfDtype(attr_value), int
                )
                and attr_value in oneflow.dtypes()
            )
            attribute.at_data_type = oneflow_api.deprecated.GetProtoDtype4OfDtype(
                attr_value
            )
        elif attr_type == attr_value_pb.kAtListInt32:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_list_int32.val[:] = list(attr_value)
        elif attr_type == attr_value_pb.kAtListInt64:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_list_int64.val[:] = list(attr_value)
        elif attr_type == attr_value_pb.kAtListFloat:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, float) for x in attr_value)
            attribute.at_list_float.val[:] = list(attr_value)
        elif attr_type == attr_value_pb.kAtListDataType:
            assert isinstance(attr_value, (tuple, list))
            assert all(
                isinstance(oneflow_api.deprecated.GetProtoDtype4OfDtype(x), int)
                and x in oneflow.dtypes()
                for x in attr_value
            )
            attribute.at_list_data_type.val[:] = list(
                [oneflow_api.deprecated.GetProtoDtype4OfDtype(x) for x in attr_value]
            )
        elif attr_type == attr_value_pb.kAtListShape:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, tuple) or isinstance(x, list) for x in attr_value)
            for i in range(len(attr_value)):
                shape = shape_util.ShapeProto()
                shape.dim[:] = list(attr_value[i])
                attribute.at_list_shape.val.append(shape)
        elif attr_type == attr_value_pb.kAtListString:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, str) for x in attr_value)
            attribute.at_list_string.val[:] = list(attr_value)
        else:
            raise ValueError("Invalid op attribute type {}".format(attr_type))

        self.user_op_.op_conf_.user_conf.attr[attr_name].CopyFrom(attribute)
        return self


@oneflow_export("user_op_module_builder")
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
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_type_name
    return UserOpModuleBuilder(LazyUserOpModule, op_name, op_type_name)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def eager_logical_user_op_module_builder(op_type_name):
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
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
                self.op_conf, oneflow_api.oneflow.core.operator.op_conf.OperatorConf
            ):
                cfg_op_conf = oneflow_api.deprecated.MakeOpConfByString(
                    str(self.op_conf)
                )
            self.set_opkernel_object(builder.NewOpKernelObject(cfg_op_conf))

        vm_util.LogicalRun(BuildInstruction)

    def InferAndTryRun(self):
        assert hob.in_global_mode(None)
        interpret_util.OpKernelForward(self.op_conf, self.opkernel_object)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.EagerLogicalBlob(lbi)


@oneflow_export("consistent_user_op_module_builder")
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
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
    op_name = name_scope.GetJobNameScopePrefix(job_name) + op_type_name
    return UserOpModuleBuilder(LazyConsistentUserOpModule, op_name, op_type_name)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def eager_consistent_user_op_module_builder(op_type_name):
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
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
                self.op_conf, oneflow_api.oneflow.core.operator.op_conf.OperatorConf
            ):
                cfg_op_conf = oneflow_api.deprecated.MakeOpConfByString(
                    str(self.op_conf)
                )
            self.set_opkernel_object(builder.NewOpKernelObject(cfg_op_conf))

        vm_util.LogicalRun(BuildInstruction)

    def InferAndTryRun(self):
        assert hob.in_global_mode(None)
        interpret_util.OpKernelConsistentForward(self.op_conf, self.opkernel_object)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.EagerLogicalBlob(lbi)
