import oneflow.oneflow_internal as oneflow_internal
from oneflow.python.oneflow_export import oneflow_export
from google.protobuf import text_format
from oneflow.python.framework.job_build_and_infer_error import JobBuildAndInferError
import oneflow.core.common.error_pb2 as error_util
import oneflow as flow
import traceback

@oneflow_export("optimizer.SGD")
class SGD(oneflow_internal.OptimizerBase):
    def __init__(self):
        oneflow_internal.OptimizerBase.__init__(self)

    def Build(self, name, a):
        try:
            self.DoBuild(name, a)
        except Exception:
            traceback.print_exc()

    def DoBuild(self, name, a):
        print(name)
        x = flow.get_variable("v1", shape=(10,), dtype=flow.float32, initializer=flow.random_uniform_initializer())
        y = flow.math.relu(x) + 1
        print(a)
        

@oneflow_export("register_optimizer")
def register_optimizer(name, optimizer):
    error_str = oneflow_internal.RegisterOptimizer(name, optimizer)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
