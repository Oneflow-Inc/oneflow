import oneflow as flow
import numpy as np
import torch

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
flow.init(config)

def AddJob(a = flow.val((5,2)), b = flow.val((5,2))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(5).data_part_num(1).default_data_type(flow.float)
    return flow.keras.maths.add(a, b, activation='sigmoid')

flow.add_job(AddJob)

a = np.arange(-5,5).reshape((5,2)).astype(np.float32)
b = np.arange(-5,5).reshape((5,2)).astype(np.float32)

with flow.Session() as sess:
    x = sess.run(AddJob, a, b).get()

y = torch.add(torch.Tensor(a), torch.Tensor(b))
y = torch.sigmoid(y)

result = np.isclose(np.array(x), y.numpy(), rtol=1e-03, atol=1e-05)
for i in result.ravel():
    assert i, "the add test is wrong!"
