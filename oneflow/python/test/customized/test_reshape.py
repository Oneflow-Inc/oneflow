import oneflow as flow
import numpy as np
import oneflow.core.framework.user_op_attr_pb2 as user_op_attr_util

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

def test_reshape(x, shape, name):
    return flow.user_op_builder(name).Op("TestReshape").Input("in",[x]) \
            .SetAttr("shape", shape, user_op_attr_util.UserOpAttrType.kAtShape) \
            .Build().RemoteBlobList()

@flow.function
def ReshapeJob(x = flow.input_blob_def((10, 2))):
    return test_reshape(x, [5,4], "xx_test_reshape")

index = [2.22, -1, 0, 1.1, 2]
data = []
for i in index: data.append(np.ones((10, 2,), dtype=np.float32) * i)
for x in data:  print(ReshapeJob(x).get())
