import oneflow as flow
import numpy as np
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.common.data_type_pb2 as data_type_conf_util

config = flow.ConfigProtoBuilder()
config.gpu_device_num(2)
config.grpc_use_no_signal()
flow.init(config)

def UpdateVariable(x, scope_name, enable_all_reduce_group = True):
    dl_net = flow.deprecated.get_cur_job_dlnet_builder()
    with dl_net.VariableScope(scope_name):
        job_conf = flow.get_cur_job_conf_builder()
        job_conf.batch_size(1).data_part_num(1).default_data_type(flow.float)
        job_conf.train_conf()
        job_conf.train_conf().primary_lr = 0.01
        job_conf.train_conf().num_of_batches_in_snapshot = 100
        job_conf.train_conf().model_update_conf.naive_conf.SetInParent()
        job_conf.train_conf().loss_lbn.extend([scope_name + "-loss_op/out"])
        job_conf.enable_all_reduce_group(enable_all_reduce_group)
        initializer = op_conf_util.InitializerConf()
        initializer.constant_conf.value = 1
        w = flow.get_variable(name = scope_name + '-w', shape = (10, ), dtype = flow.float, initializer = initializer)
        c = dl_net.BiasAdd(x, w)
        # return flow.keras.activations.tanh(c) 
        return dl_net.ReduceMean(c, name="loss_op")

input_blob_def = flow.input_blob_def((2, 10), dtype=flow.float)
def OneDeviceUpdateVariable(x = input_blob_def):
    with flow.device_prior_placement("gpu", "0:0"):
        return UpdateVariable(x, "one-device")
flow.add_job(OneDeviceUpdateVariable)

def TwoDeviceUpdateVariable(x = input_blob_def):
    with flow.device_prior_placement("gpu", "0:0-1"):
        return UpdateVariable(x, "two-device")
flow.add_job(TwoDeviceUpdateVariable)

def DisableAllReduceGroupUpdateVariable(x = input_blob_def):
    with flow.device_prior_placement("gpu", "0:0-1"):
        return UpdateVariable(x, "disable-all-reduce-group", False)
flow.add_job(DisableAllReduceGroupUpdateVariable)

status = flow.train.CheckPoint().restore()
with flow.Session() as sess:
    status.initialize_or_restore(session = sess)
    print "-------- one device --------"
    for i in range(10):
        x = sess.run(OneDeviceUpdateVariable, np.ones((2, 10), dtype=np.float32)).get()
        print(x)
    print "-------- two device --------"
    for i in range(10):
        x = sess.run(TwoDeviceUpdateVariable, np.ones((2, 10), dtype=np.float32)).get()
        print(x)
    print "-------- disable all reduce group --------"
    for i in range(10):
        x = sess.run(DisableAllReduceGroupUpdateVariable, np.ones((2, 10), dtype=np.float32)).get()
        print(x)
