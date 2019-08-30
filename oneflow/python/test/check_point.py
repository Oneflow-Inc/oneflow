import oneflow as flow
import numpy as np

#config = flow.ConfigProtoBuilder()
#config.gpu_device_num(1)
#flow.init(config)

def ReluJob(x = flow.input_blob_def((10,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.keras.activations.relu(x)
flow.add_job(ReluJob)

ckp = flow.train.CheckPoint()
with flow.Session() as sess:
    x = np.ones((10,), dtype=np.float32)
    ckp.save(session = sess)
    print(sess.run(ReluJob, x).get())
