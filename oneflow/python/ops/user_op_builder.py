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

class UserOpConfWrapper(object):
    def __init__(self, op_name):
        self.op_conf_ = op_conf_util.OperatorConf()
        self.op_conf_.name = op_name

    def RemoteBlobList(self):
        remote_blob_list = []
        compile_context.CurJobAddOp(self.op_conf_)
        for k in self.op_conf_.user_conf.output:
            for i in range(len(self.op_conf_.user_conf.output[k].s)):
                lbi = logical_blob_id_util.LogicalBlobId()
                lbi.op_name = self.op_conf_.name
                lbi.blob_name = k + '_' + str(i)
                remote_blob_list.append(remote_blob_util.RemoteBlob(lbi))
        # cclog:
        print("cclog: op_name: ", self.op_conf_.name, "\n")
        print("cclog: op_type_name: ", self.op_conf_.user_conf.op_type_name, "\n")
        print("cclog: input: \n")
        for k in self.op_conf_.user_conf.input:
            for i in range(len(self.op_conf_.user_conf.input[k].s)):
                print("cclog: input_name: ", k, " lbn: ", self.op_conf_.user_conf.input[k].s[i])
        print("cclog: output: \n")
        for k in self.op_conf_.user_conf.output:
            for i in range(len(self.op_conf_.user_conf.output[k].s)):
                print("cclog: output_name: ", k, " lbn: ", self.op_conf_.user_conf.output[k].s[i])
        print("cclog: attr: \n")
        for k in self.op_conf_.user_conf.attr:
            print("cclog: attr_name: ", k)
            # TODO(); print attr val
        print("cclog: remote blob: \n")
        for blob in remote_blob_list:
            print("cclog: blob_name: ", blob.logical_blob_name, " shape: ", blob.shape)
        print("cclog Done: \n")
        # cclog Done;
        return tuple(remote_blob_list)

@oneflow_export('user_op_builder')
class UserOpConfWrapperBuilder(object):
    def __init__(self, op_name):
        self.user_op_ = UserOpConfWrapper(op_name)

    def Build(self):
        assert self.user_op_.op_conf_.user_conf.op_type_name is not ""
        self.user_op_.op_conf_ = \
            job_builder.CurCtxCheckAndCompleteUserOpConf(self.user_op_.op_conf_)
        print("cclog: op conf: ", self.user_op_.op_conf_)
        return self.user_op_

    def Op(self, op_type_name):
        self.user_op_.op_conf_.user_conf.op_type_name = op_type_name
        return self

    def Input(self, input_name, input_blob_list):
        assert isinstance(input_blob_list, (tuple, list))
        for input_blob in input_blob_list:
            # assert type(input_blob) is blob_desc.BlobDesc
            self.user_op_.op_conf_.user_conf.input[input_name].s.append(input_blob.logical_blob_name)
        return self

    def Output(self, output_name, num = 1):
        assert type(num) is int and num >= 1
        out_lbns = []
        for i in range(num):
            lbn = self.user_op_.op_conf_.name + '/' + output_name + '_' + str(i)
            out_lbns.append(lbn)
        self.user_op_.op_conf_.user_conf.output[output_name].s[:] = out_lbns
        return self

    def SetAttr(self, attr_name, attr_value, attr_type):
        attribute = user_op_attr_util.UserOpAttrVal()
        assert type(attr_name) is str
        assert type(attr_type) is int
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
            assert all(isinstance(x, int) for x in attr_value)
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
        self.user_op_.op_conf_.user_conf.attr[attr_name].CopyFrom(attribute)
        return self

