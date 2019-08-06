import oneflow as flow
import numpy as np
import torch
import math

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
flow.init(config)

def TanhJob(x = flow.input_blob_def((10,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.keras.activations.tanh(x)

flow.add_job(TanhJob)

x = np.array(range(-5,5), dtype=np.float32)


with flow.Session() as sess:
    a = sess.run(TanhJob, x).get()

b = torch.tanh(torch.Tensor(x))

result = np.isclose(np.array(a), b.numpy(), rtol=1e-03, atol=1e-05)
for i in result:
    assert i, "the tanh test is wrong!"
