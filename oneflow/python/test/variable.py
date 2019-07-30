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
    job_conf.batch_size(5).data_part_num(1).default_data_type(flow.float)
    return flow.keras.variables.variable((5,2), name="v1")

with flow.Session(jobs) as sess:
    x = sess.run(VariableJob).get()

print(x)

