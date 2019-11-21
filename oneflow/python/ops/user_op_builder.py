from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.job_builder as job_builder
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.framework.user_op_attr_pb2 as user_op_attr_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow
from oneflow.python.oneflow_export import oneflow_export

class UserOpWrapper(object):
    def __init__(self, op_name):
        self.op_conf_ = op_conf_util.OperatorConf()
        self.user_conf_ = self.op_conf_.user_conf
        self.op_conf_.name = op_name

    @property
    def op_type_name(self): return self.user_conf_.op_type_name

    @property
    def user_conf(self): return self.user_conf_

    @property
    def op_conf(self): return self.op_conf_

    def RemoteBlobList(self):
        remote_blob_list = []
        compile_context.CurJobAddOp(self.op_conf_)
        for k in self.user_conf_.output:
            for i in range(self.user_conf_.output[k].s):
                lbi = logical_blob_id_util.LogicalBlobId()
                lbi.op_name = self.op_conf_.name
                lbi.blob_name = k + '_' + str(i)
                remote_blob_list.append(remote_blob_util.RemoteBlob(lbi))
        return tuple(remote_blob_list)

class UserOpWrapperBuilder(object):
    def __init__(self, op_name):
        self.user_op_ = UserOpWrapper(op_name)

    def Build(self):
        assert self.user_op_.op_type_name is not ""
        self.user_op_.op_conf = \
            job_builder.CurCtxAddDefaultValueAndCheckValid4UserOp(self.user_op_.op_conf)
        return self.user_op_

    def Op(self, op_type_name):
        # self.op_registration_val_ = c_api_util.LookUpInOpRegistry(op_type_name)
        self.user_op_.op_type_name = op_type_name
        return self

    def Input(self, input_name, input_blob_list):
        assert isinstance(input_blob_list, (tuple, list))
        for input_blob in input_blob_list
          assert type(input_blob) is blob_desc.BlobDesc
          self.user_op_.user_conf.input[input_name].s.append(input_blob.logical_blob_name)
        return self

    def Output(self, output_name, num = 1):
        assert type(num) is int and num >= 1
        out_lbns = []
        for i in range(num):
            lbn = self.op_conf_.name + '/' output_name + '_' + str(i)
            out_lbns.append(lbn)
        self.user_op_.user_conf.output[output_name].s[:] = out_lbns
        return self

    def SetAttr(self, attr_name, attr_value, attr_type):
        attribute = user_op_attr_util.UserOpAttrVal()
        assert type(attr_name) is str
        assert type(attr_type) is user_op_attr_util.UserOpAttrType
        if attr_type == user_op_attr_util.UserOpAttrType.kAtInt32:
            assert type(attr_value) is int
            attribute.at_int32 = attr_value
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtInt64:
            assert type(attr_value) is int
            attribute.at_int64 = attr_value
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtBool:
            assert type(attr_value) is bool
            attribute.at_bool = attr_value
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtFloat:
            assert type(attr_value) is float
            attribute.at_float = attr_value
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtDouble:
            assert type(attr_value) is float
            attribute.at_double = attr_value
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtString:
            assert type(attr_value) is str
            attribute.at_string = attr_value
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtShape:
            assert isinstance(attr_value, (tuple, list))
            attribute.at_shape.dim[:] = list(attr_value)
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtListInt32:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_list_int32.val[:] = list(attr_value)
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtListInt64:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_list_int64.val[:] = list(attr_value)
        elif attr_type == user_op_attr_util.UserOpAttrType.kAtListFloat:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, float) for x in attr_value)
            attribute.at_list_float.val[:] = list(attr_value)
        else:
            assert False, "Unknow op attribute type: {}".format(attr_type)
        self.user_op_.user_conf.attr[attr_name] = attribute
        return self

