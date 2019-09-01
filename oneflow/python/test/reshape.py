import oneflow as flow
import numpy as np

config = flow.ConfigProtoBuilder()
config.gpu_device_num(2)
config.ctrl_port(12322)
flow.init(config)


def ReshapeJob0(x=flow.input_blob_def((10, 20, 20))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.reshape(x, (200, 20))


flow.add_job(ReshapeJob0)

with flow.Session() as sess:
    random_array = np.random.rand(10, 20, 20).astype(np.float32)
    np.array_equal(sess.run(ReshapeJob0, random_array).get(),
                   np.reshape(random_array, (200, 20)))
