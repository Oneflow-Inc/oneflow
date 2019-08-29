import oneflow as flow
import numpy as np

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
flow.init(config)


def MaxPool2DTestJob(x=flow.input_blob_def((10, 1000, 1000, 3))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return flow.keras.pooling.MaxPool2D(
        x,
        pool_size=[10, 10],
        strides=[10, 10],
        padding="valid",
        data_format="channels_last",
    )


flow.add_job(MaxPool2DTestJob)

x = np.random.random_sample((10, 1000, 1000, 3)).astype(np.float32)

with flow.Session() as sess:
    sess.run(MaxPool2DTestJob, x).get()
