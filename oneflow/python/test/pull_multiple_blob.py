import oneflow as flow
import numpy as np

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
config.grpc_use_no_signal()
flow.init(config)

def TestNet(x=flow.input_blob_def((1,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(flow.float)
    dlnet = flow.deprecated.get_cur_job_dlnet_builder()
    return (x, x)
flow.add_job(TestNet)

with flow.Session() as sess:
    x = np.array([1], dtype=np.float32)
    fetched = sess.run(TestNet, x).get()
    print(fetched)
