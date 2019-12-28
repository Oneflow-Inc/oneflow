import oneflow as flow
import numpy as np
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.common.data_type_pb2 as data_type_conf_util

flow.config.gpu_device_num(2)

def UpdateVariable(x, scope_name):
    with flow.variable_scope(scope_name):
        w = flow.get_variable(name = scope_name + '-w', shape = (10, ),
            dtype = flow.float, initializer = flow.constant_initializer(value=1.0))
        c = flow.nn.bias_add(x, w)
        # return flow.keras.activations.tanh(c) 
        loss = flow.math.reduce_mean(c, name="loss_op")
        flow.losses.add_loss(loss)
        return loss

input_blob_def = flow.FixedTensorDef((2, 10), dtype=flow.float)
func_config = flow.FunctionConfig()
func_config.train.primary_lr(0.01)
func_config.train.model_update_conf(dict(naive_conf={}))
func_config.enable_all_reduce_group(True)
@flow.function(func_config)
def OneDeviceUpdateVariable(x = input_blob_def):
    with flow.device_prior_placement("gpu", "0:0"):
        return UpdateVariable(x, "one-device")

@flow.function(func_config)
def TwoDeviceUpdateVariable(x = input_blob_def):
    with flow.device_prior_placement("gpu", "0:0-1"):
        return UpdateVariable(x, "two-device")

func_config.enable_all_reduce_group(True)
@flow.function(func_config)
def DisableAllReduceGroupUpdateVariable(x = input_blob_def):
    with flow.device_prior_placement("gpu", "0:0-1"):
        return UpdateVariable(x, "disable-all-reduce-group")

print "-------- one device --------"
for i in range(10):
    x = OneDeviceUpdateVariable(np.ones((2, 10), dtype=np.float32)).get()
    print(x)
print "-------- two device --------"
for i in range(10):
    x = TwoDeviceUpdateVariable(np.ones((2, 10), dtype=np.float32)).get()
    print(x)
print "-------- disable all reduce group --------"
for i in range(10):
    x = DisableAllReduceGroupUpdateVariable(np.ones((2, 10), dtype=np.float32)).get()
    print(x)
