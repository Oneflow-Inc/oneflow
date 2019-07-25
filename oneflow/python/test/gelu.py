import oneflow as flow
import numpy as np
import torch
import math

jobs = []
@flow.append_func_to_list(jobs)
def GeluJob(x = flow.val((10,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.keras.activations.gelu(x)

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)


x = np.ones((10,), dtype=np.float32)

with flow.Session(jobs, config) as sess:
    a = sess.run(GeluJob, x).get()

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))
x = torch.tensor(x)
b = gelu(x)

result = np.isclose(np.array(a), b.numpy(), rtol=1e-03, atol=1e-05)
for i in result:
    assert i, "the test is wrong!"
