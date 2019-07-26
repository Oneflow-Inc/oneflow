import oneflow as flow
import numpy as np


config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
flow.init(config)

jobs = []
@flow.append_func_to_list(jobs)
def ReluJob(x = flow.val((10,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.keras.activations.relu(x)

with flow.Session(jobs) as sess:
    index = [-2, -1, 0, 1, 2]
    data = []
    for i in index: data.append(np.ones((10,), dtype=np.float32) * i)
    for x in data:  print(sess.run(ReluJob, x).get())
    
