import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.distribute as distribute
import oneflow.python.framework.hob as hob
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.framework.user_op_attr_pb2 as user_op_attr_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.common.shape_pb2 as shape_util
import oneflow
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.hob as hob
import oneflow.python.experimental.name_scope as name_scope
import oneflow.core.vm.instruction_pb2 as instr_util
import oneflow.core.eager.eager_symbol_pb2 as eager_symbol_util
import oneflow.python.vm.id_util as id_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.job_conf_ctx as job_conf_ctx
import oneflow.python.eager.eager_blob_util as eager_blob_util
import oneflow.python.lib.core.enable_if as enable_if
import random
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.eager.blob_register as blob_register_util
import traceback

blob_register = blob_register_util.GetDefaultBlobRegister()


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


@oneflow_export("user_op_builder")
def api_user_op_builder(op_name):
    """Build a wrapper of user op.

    Args:
        op_name (str): name of new user op

    Returns:
        UserOpConfBuilder: `UserOpConfBuilder` object used to build a wrapper of user op.
    
    Example::

        def myargmax(
            input: remote_blob_util.BlobDef
        ) -> remote_blob_util.BlobDef:
            return (
            flow.user_op_builder("myargmax")
            .Op("argmax")
            .Input("in", [input])
            .Output("out")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
            )   

    """
    api = enable_if.unique(
        [
            lazy_user_op_builder,
            eager_logical_user_op_builder,
            eager_physical_user_op_builder,
        ]
    )
    return api(op_name)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def lazy_user_op_builder(op_name):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return UserOpConfBuilder(job_name, op_name, LazyUserOp)


class LazyUserOp(UserOp):
    def __init__(self, op_name):
        UserOp.__init__(self, op_name)

    def InferAndTryRun(self):
        compile_context.CurJobAddOp(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.RemoteBlob(lbi)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def eager_logical_user_op_builder(op_name):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return UserOpConfBuilder(job_name, op_name, EagerLogicalUserOp)


class EagerLogicalUserOp(UserOp):
    def __init__(self, op_name):
        UserOp.__init__(self, op_name)

    def InferAndTryRun(self):
        interpret_util.Forward(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.EagerLogicalBlob(lbi)


in_physical_placement = hob.env_initialized & hob.is_current_placement_physical


@enable_if.condition(hob.in_normal_mode & in_physical_placement)
def eager_physical_user_op_builder(op_name):
    job_name = job_conf_ctx.CurrentJobConf().job_name
    return UserOpConfBuilder(job_name, op_name, EagerPhysicalUserOp)


class EagerPhysicalUserOp(UserOp):
    def __init__(self, op_name):
        UserOp.__init__(self, op_name)

    def InferAndTryRun(self):
        self.op_conf_.scope_symbol_id = oneflow.scope.current_scope().symbol_id
        op_attribute = c_api_util.GetOpAttribute4OpConf(self.op_conf_)

        def BuildInstruction(builder):
            with blob_register.BnInOp2BlobObjectScope(
                op_attribute
            ) as bn_in_op2blob_object:
                parallel_conf = oneflow.placement.current_scope().default_parallel_conf
                builder.StatelessCall(
                    op_attribute,
                    parallel_conf,
                    bn_in_op2blob_object=bn_in_op2blob_object,
                )

        vm_util.PhysicalRun(BuildInstruction)
        return self

    def MakeRemoteBlob(self, lbi):
        return eager_blob_util.EagerPhysicalBlob("%s/%s" % (lbi.op_name, lbi.blob_name))


@oneflow_export("consistent_user_op_builder")
def consistent_user_op_builder(op_name):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return UserOpConfBuilder(job_name, op_name, ConsistentUserOp)


class ConsistentUserOp(UserOp):
    def __init__(self, op_name):
        UserOp.__init__(self, op_name)

    def InferAndTryRun(self):
        interpret_util.ConsistentForward(self.op_conf_)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.RemoteBlob(lbi)


def NonTraceableEagerLogicalUserOpBuilder(op_name):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return UserOpConfBuilder(job_name, op_name, NonTraceableEagerLogicalUserOp)


class NonTraceableEagerLogicalUserOp(UserOp):
    def __init__(self, op_name):
        UserOp.__init__(self, op_name)

    def InferAndTryRun(self):
        self.op_conf_.scope_symbol_id = oneflow.scope.current_scope().symbol_id
        op_attribute = c_api_util.GetOpAttribute4OpConf(self.op_conf_)

        def BuildInstruction(builder):
            get_scope = blob_register.BnInOp2BlobObjectScope
            with get_scope(op_attribute) as bn_in_op2blob_object:
                parallel_conf = oneflow.placement.current_scope().default_parallel_conf
                builder.StatelessCall(
                    op_attribute,
                    parallel_conf,
                    bn_in_op2blob_object=bn_in_op2blob_object,
                )

        vm_util.LogicalRun(BuildInstruction)
        return self

    def MakeRemoteBlob(self, lbi):
        return remote_blob_util.EagerLogicalBlob(lbi)


class UserOpConfBuilder(object):
    def __init__(self, job_name, op_name, user_op_class):
        name_scope_prefix = name_scope.GetJobNameScopePrefix(job_name)
        self.user_op_ = user_op_class(name_scope_prefix + op_name)

    def Build(self):
        """Build op when in/output and other attribute set up.

        Returns:
            self
        """
        assert self.user_op_.op_conf_.user_conf.op_type_name != ""
        self.user_op_.op_conf_ = c_api_util.CheckAndCompleteUserOpConf(
            self.user_op_.op_conf_
        )
        return self.user_op_

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
        for input_blob in input_blob_list:
            # assert type(input_blob) is blob_desc.BlobDesc
            self.user_op_.op_conf_.user_conf.input[input_name].s.append(
                input_blob.unique_name
            )
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
                        """WARNING: Argument 'attr_type_name' of UserOpConfBuilder.Attr has been deprecated. Please remove it.
            For instance:
                -     .Attr("out_num", out_num, "AttrTypeInt64")
                +     .Attr("out_num", out_num)
                        """
                    )
            print(traceback.format_stack()[-2])

        attribute = user_op_attr_util.UserOpAttrVal()
        assert isinstance(attr_name, str)
        attr_type = c_api_util.GetUserOpAttrType(
            self.user_op_.op_conf_.user_conf.op_type_name, attr_name
        )
        if attr_type == user_op_attr_util.kAtInt32:
            assert isinstance(attr_value, int)
            attribute.at_int32 = attr_value
        elif attr_type == user_op_attr_util.kAtInt64:
            assert isinstance(attr_value, int)
            attribute.at_int64 = attr_value
        elif attr_type == user_op_attr_util.kAtBool:
            assert isinstance(attr_value, bool)
            attribute.at_bool = attr_value
        elif attr_type == user_op_attr_util.kAtFloat:
            assert isinstance(attr_value, float)
            attribute.at_float = attr_value
        elif attr_type == user_op_attr_util.kAtDouble:
            assert isinstance(attr_value, float)
            attribute.at_double = attr_value
        elif attr_type == user_op_attr_util.kAtString:
            assert isinstance(attr_value, str)
            attribute.at_string = attr_value
        elif attr_type == user_op_attr_util.kAtShape:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_shape.dim[:] = list(attr_value)
        elif attr_type == user_op_attr_util.kAtDataType:
            assert isinstance(attr_value, int) and attr_value in oneflow.dtypes
            attribute.at_data_type = attr_value
        elif attr_type == user_op_attr_util.kAtListInt32:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_list_int32.val[:] = list(attr_value)
        elif attr_type == user_op_attr_util.kAtListInt64:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) for x in attr_value)
            attribute.at_list_int64.val[:] = list(attr_value)
        elif attr_type == user_op_attr_util.kAtListFloat:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, float) for x in attr_value)
            attribute.at_list_float.val[:] = list(attr_value)
        elif attr_type == user_op_attr_util.kAtListDataType:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, int) and x in oneflow.dtypes for x in attr_value)
            attribute.at_list_data_type.val[:] = list(attr_value)
        elif attr_type == user_op_attr_util.kAtListShape:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, tuple) or isinstance(x, list) for x in attr_value)
            for i in range(len(attr_value)):
                shape = shape_util.ShapeProto()
                shape.dim[:] = list(attr_value[i])
                attribute.at_list_shape.val.append(shape)
        elif attr_type == user_op_attr_util.kAtListString:
            assert isinstance(attr_value, (tuple, list))
            assert all(isinstance(x, str) for x in attr_value)
            attribute.at_list_string.val[:] = list(attr_value)
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

        return self.Attr("has_seed", (seed is not None)).Attr("seed", seed)
