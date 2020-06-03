from google.protobuf import text_format
import traceback
import oneflow.oneflow_internal as oneflow_internal
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.job_build_and_infer_error import JobBuildAndInferError
import oneflow.core.common.error_pb2 as error_util
import oneflow.python.framework.hob as hob
import oneflow.python.framework.runtime_mode as rt_mode
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.placement_pb2 as placment_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.core.job.job_pb2 import TrainConf
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_context
import oneflow.python.framework.remote_blob as remote_blob_util

def build_sgd(var, var_diff, lr, var_op_conf, parallel_conf):
    assert hob.in_global_mode(), "must build optimizer in {}, current mode: {}".format(
        rt_mode.GLOBAL_MODE, rt_mode.CurrentMode()
    )
    flow.assign(var, var - var_diff * lr)
    # x = flow.constant(-1, shape=(1,2), dtype=flow.float32)
    y = flow.math.relu(x, name="relu2020") + 1
    print(a)

def lr_lbn_from_train_conf(train_conf):
    if var_op_conf.variable_conf.model_name == "weight":
        return train_conf.primary_lr_lbn
    elif var_op_conf.variable_conf.model_name == "bias":
        return train_conf.secondary_lr_lbn
    else:
        return train_conf.primary_lr_lbn 
@oneflow_export("optimizer.SGD")
class SGD(oneflow_internal.OptimizerBase):
    def __init__(self):
        oneflow_internal.OptimizerBase.__init__(self)

    def Build(self, var_op_conf_txt, parallel_conf_txt, diff_lbi_of_var_out_txt, train_conf_txt):
        try:
            with rt_mode.ModeScope(rt_mode.GLOBAL_MODE):
                print(var_op_conf_txt, parallel_conf_txt, diff_lbi_of_var_out_txt)
                var_op_conf = text_format.Parse(var_op_conf_txt, op_conf_util.OperatorConf())
                parallel_conf = text_format.Parse(parallel_conf_txt, placment_util.ParallelConf())
                diff_lbi = text_format.Parse(diff_lbi_of_var_out_txt, logical_blob_id_util.LogicalBlobId())
                train_conf = text_format.Parse(train_conf_txt, TrainConf())
                
                job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
                sess = session_context.GetDefaultSession()
                var = sess.TryGetVariableBlobOfJobFromStash(job_name, var_op_conf.name)
                var_diff = remote_blob_util.RemoteBlob(diff_lbi)
                lr_lbn = lr_lbn_from_train_conf(train_conf)
                print(lr_lbn)
                # build_sgd(var, var_diff, var_op_conf, parallel_conf, train_conf)
        except Exception:
            traceback.print_exc()


@oneflow_export("register_optimizer")
def register_optimizer(name, optimizer):
    error_str = oneflow_internal.RegisterOptimizer(name, optimizer)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
