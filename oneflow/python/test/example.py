import oneflow as flow
import numpy as np

jobs = []
@flow.append_func_to_list(jobs)
def DemoJob(x = flow.val((10,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return x

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)

with flow.Session(jobs, config) as sess:
    data = []
    for i in range(5): data.append(np.ones((10,), dtype=np.float32) * i)
    for x in data: print sess.run(DemoJob, x).get()
    for x in sess.map(DemoJob, data).get(): print x
