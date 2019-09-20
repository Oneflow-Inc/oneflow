import oneflow as flow
import numpy as np

@flow.function
def AddJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(5).default_data_type(flow.float)
    a + b
    return a + b + b


x = np.random.rand(5, 2).astype(np.float32)
y = np.random.rand(5, 2).astype(np.float32)
z = None

z = AddJob(x, y).get()

print (np.array_equal(z, x + y + y))
