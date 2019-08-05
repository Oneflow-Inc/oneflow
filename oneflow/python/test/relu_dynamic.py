import oneflow as flow
import numpy as np


config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
config.ctrl_port(12311)
config.grpc_use_no_signal()
flow.init(config)

def ReluJob(x = flow.val(shape = (10,), is_dynamic=True)):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.keras.activations.relu(x)
flow.add_job(ReluJob)

with flow.Session() as sess:
    print(sess.run(ReluJob, np.array([-2, 5, -5, 2], dtype=np.float32)).get())
    print(sess.run(ReluJob, np.array([2, 0, 0, 5, -7, 9], dtype=np.float32)).get())
    print(sess.run(ReluJob, np.array([1], dtype=np.float32)).get())

    '''
    output:
      [0. 5. 0. 2.]
      [2. 0. 0. 5. 0. 9.]
      [1.]
    '''

