import oneflow as flow
import numpy as np
import torch
import math

#config = flow.ConfigProtoBuilder()
#config.gpu_device_num(1)
#flow.init(config)

def SigmoidJob(x = flow.input_blob_def((10,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.keras.activations.sigmoid(x)

flow.add_job(SigmoidJob)

x = np.array(range(-5,5), dtype=np.float32)

with flow.Session() as sess:
    a = sess.run(SigmoidJob, x).get()

b = torch.sigmoid(torch.Tensor(x))

result = np.isclose(np.array(a), b.numpy(), rtol=1e-03, atol=1e-05)
for i in result:
    assert i, "the sigmoid test is wrong!"
