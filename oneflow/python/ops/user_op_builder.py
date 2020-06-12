import random

import oneflow as flow
import oneflow.core.common.shape_pb2 as shape_util
import oneflow.core.framework.user_op_attr_pb2 as user_op_attr_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.experimental.name_scope as name_scope
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.distribute as distribute
import oneflow.python.framework.hob as hob
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export


class UserOp(object):
    def __init__(self, op_name):
        self.op_conf_ = op_conf_util.OperatorConf()
        self.op_conf_.name = op_name
        self.output_arg_key_list_ = []

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
                remote_blob_list.append(self.MakeRemoteBlob(lbi))

        return tuple(remote_blob_list)

    def SoleOutputBlob(self):
        blobs = self.RemoteBlobList()
        assert len(blobs) == 1
        return blobs[0]


class LazyUserOp(UserOp):
    def __init__(self, op_name):
        UserOp.__init__(self, op_name)

    def InferAndTryRun(self):
        compile_context.CurJobAddOp(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.RemoteBlob(lbi)


@enable_if.condition(hob.in_global_mode)
def lazy_user_op_builder(op_name):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return UserOpConfBuilder(job_name, op_name, LazyUserOp)


@oneflow_export("user_op_builder")
def api_user_op_builder(op_name):
    return enable_if.unique(lazy_user_op_builder)(op_name)


class ConsistentUserOp(UserOp):
    def __init__(self, op_name):
        UserOp.__init__(self, op_name)

    def InferAndTryRun(self):
        compile_context.CurJobAddConsistentOp(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.RemoteBlob(lbi)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def consistent_user_op_builder(op_name):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return UserOpConfBuilder(job_name, op_name, ConsistentUserOp)


@oneflow_export("consistent_user_op_builder")
def api_consistent_user_op_builder(op_name):
    return enable_if.unique(consistent_user_op_builder)(op_name)


class UserOpConfBuilder(object):
    def __init__(self, job_name, op_name, user_op_class):
        name_scope_prefix = name_scope.GetJobNameScopePrefix(job_name)
        self.user_op_ = user_op_class(name_scope_prefix + op_name)

    def Build(self):
        assert self.user_op_.op_conf_.user_conf.op_type_name != ""
        self.user_op_.op_conf_ = c_api_util.CheckAndCompleteUserOpConf(
            self.user_op_.op_conf_
        )
        return self.user_op_

    def Op(self, op_type_name):
        self.user_op_.op_conf_.user_conf.op_type_name = op_type_name
        return self

    def Input(self, input_name, input_blob_list):
        assert isinstance(input_blob_list, (tuple, list))
        for input_blob in input_blob_list:
            # assert type(input_blob) is blob_desc.BlobDesc
            self.user_op_.op_conf_.user_conf.input[input_name].s.append(
                input_blob.unique_name
            )
        return self

    def Output(self, output_name, num=1):
        assert isinstance(num, int) and num >= 1
        out_lbns = []
        for i in range(num):
            lbn = "{}/{}_{}".format(self.user_op_.op_conf_.name, output_name, i)
            out_lbns.append(lbn)
        self.user_op_.op_conf_.user_conf.output[output_name].s[:] = out_lbns
        self.user_op_.output_arg_key_list_.append(output_name)
        return self

    def Attr(self, attr_name, attr_value, attr_type):
        attribute = user_op_attr_util.UserOpAttrVal()
        assert isinstance(attr_name, str)
        assert isinstance(attr_type, str)
        if attr_type == "AttrTypeInt32":
            assert isinstance(attr_value, int)
            attribute.at_int32 = attr_value
        elif attr_type == "AttrTypeInt64":
            assert isinstance(attr_value, int)
            attribute.at_int64 = attr_value
        elif attr_type == "AttrTypeBool":
            assert isinstance(attr_value, bool)
            attribute.at_bool = attr_value
        elif attr_type == "AttrTypeFloat":
            assert isinstance(attr_value, float)
            attribute.at_float = attr_value
        elif attr_type == "AttrTypeDouble":
            assert isinstance(attr_value, float)
            attribute.at_double = attr_value
        elif attr_type == "AttrTypeString":
            assert isinstance(attr_value, str)
            attribute.at_string = attr_value
        elif attr_type == "AttrTypeShape":
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_shape.dim[:] = list(attr_value)
        elif attr_type == "AttrTypeDataType":
            assert isinstance(attr_value, int) and attr_value in flow.dtypes
            attribute.at_data_type = attr_value
        elif attr_type == "AttrTypeListInt32":
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_list_int32.val[:] = list(attr_value)
        elif attr_type == "AttrTypeListInt64":
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_list_int64.val[:] = list(attr_value)
        elif attr_type == "AttrTypeListFloat":
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, float) for x in attr_value)
            attribute.at_list_float.val[:] = list(attr_value)
        elif attr_type == "AttrTypeListDataType":
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) and x in flow.dtypes for x in attr_value)
            attribute.at_list_data_type.val[:] = list(attr_value)
        elif attr_type == "AttrTypeListShape":
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, tuple) or isinstance(x, list) for x in attr_value)
            for i in range(len(attr_value)):
                shape = shape_util.ShapeProto()
                shape.dim[:] = list(attr_value[i])
                attribute.at_list_shape.val.append(shape)
        else:
            raise ValueError("Invalid op attribute type {}".format(attr_type))

        self.user_op_.op_conf_.user_conf.attr[attr_name].CopyFrom(attribute)
        return self

    def SetRandomSeed(self, seed=None):
        if distribute.ConsistentStrategyEnabled():
            if seed is None:
                seed = random.randint(-2147483648, 2147483647)
        elif distribute.MirroredStrategyEnabled():
            if seed is None:
                seed = -1
        else:
            raise ValueError(
                "Unknow distirbute strategy when set random seed to user op"
            )

        return self.Attr("has_seed", (seed is not None), "AttrTypeBool").Attr(
            "seed", seed, "AttrTypeInt64"
        )
