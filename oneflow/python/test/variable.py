import oneflow as flow
import numpy as np
import torch

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
flow.init(config)

jobs = []
@flow.append_func_to_list(jobs)
def VariableJob():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(flow.float)
    initializer = flow.keras.initializers.constant_int_init(value=5)
    return flow.variable((5,2), initializer = initializer, name="v1")

ckp = flow.train.CheckPoint()
status = ckp.restore()
with flow.Session(jobs) as sess:
    status.initialize_or_restore(session = sess)
    x = sess.run(VariableJob).get()

print(x)

