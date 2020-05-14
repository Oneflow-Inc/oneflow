import os
import oneflow as flow
import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("math.invert_permutation")
def invert_permutation(x, name=None):
    if name is None:
            name = id_util.UniqueStr("invert_permutation")
    return flow.user_op_builder(name).Op("invert_permutation")\
               .Input("in",[x])\
               .Output("out") \
               .Build()\
               .InferAndTryRun()\
               .RemoteBlobList()[0]
                