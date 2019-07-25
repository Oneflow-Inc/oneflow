import oneflow as flow
import numpy as np
import torch
import math

jobs = []
@flow.append_func_to_list(jobs)
def GeluJob(x = flow.val((10,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.keras.activatios.gelu(x)

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)

data = []
for i in range(5): data.append(np.ones((10,), dtype=np.float32))

with flow.Session(jobs, config) as sess:
    print("the result of oneflow is:")
    for x in data:  print(sess.run(GeluJob, x).get())



def gelu(x):
    return 0.5 * x * (1 + troch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))
x = torch.tensor(data)
print("the result of pytorch is:")
print(gelu(x))
