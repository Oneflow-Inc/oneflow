import oneflow as flow
import numpy as np

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
flow.init(config)


def AddJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(5).data_part_num(1).default_data_type(flow.float)
    a + b
    return a + b + b


flow.add_job(AddJob)

x = np.random.rand(5, 2).astype(np.float32)
y = np.random.rand(5, 2).astype(np.float32)
z = None

with flow.Session() as sess:
    z = sess.run(AddJob, x, y).get()

assert np.array_equal(z, x + y + y)
