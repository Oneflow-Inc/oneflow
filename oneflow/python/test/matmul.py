import oneflow as flow
import numpy as np
import torch

#config = flow.ConfigProtoBuilder()
#config.gpu_device_num(1)
#flow.init(config)

def MatmulJob(a = flow.input_blob_def((4, 5)), b = flow.input_blob_def((5,4))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(4).data_part_num(1).default_data_type(flow.float)
    return flow.keras.maths.matmul(a, b)

flow.add_job(MatmulJob)

a = np.arange(1,21).reshape((4,5)).astype(np.float32)
b = np.arange(1,21).reshape((5,4)).astype(np.float32)

with flow.Session() as sess:
    x = sess.run(MatmulJob, a, b).get()

y = torch.matmul(torch.Tensor(a), torch.Tensor(b))

result = np.isclose(np.array(x), y.numpy(), rtol=1e-03, atol=1e-05)
for i in result.ravel():
    assert i, "the matmul test is wrong!"
