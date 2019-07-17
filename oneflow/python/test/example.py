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
config.grpc_use_no_signal()

with flow.Session(jobs, config) as sess:
  sess.run(DemoJob, np.ones((10,), dtype=np.float32))
  print "good"

#job_set = flow.Session(jobs, config).job_set_
#from google.protobuf import text_format
#print text_format.MessageToString(job_set)
