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
  ndarray = sess.run(DemoJob, np.ones((10,), dtype=np.float32) * 10).get()
  print ndarray
  ndarray = sess.run(DemoJob, np.zeros((10,), dtype=np.float32)).get()
  print ndarray
