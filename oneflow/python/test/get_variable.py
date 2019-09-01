import oneflow as flow
import numpy as np
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.common.data_type_pb2 as data_type_conf_util

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
flow.init(config)

def GetVariableJob():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(flow.float)
    v1 = flow.get_variable(name = 'v1', shape = (5,2), dtype = data_type_conf_util.kFloat, initializer = flow.random_uniform_initializer(min=0, max=10)) 
    v2 = flow.get_variable(name = 'v1') 
    return v2 

flow.add_job(GetVariableJob)

ckp = flow.train.CheckPoint()
status = ckp.restore()
with flow.Session() as sess:
    status.initialize_or_restore(session = sess)
    x = sess.run(GetVariableJob).get()

print(x)

