import numpy as np
import tensorflow as tf
import oneflow as flow


# OneFlow
config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
flow.init(config)

def ConcatJob(
    a = flow.input_blob_def((5,2)),
    b = flow.input_blob_def((5,3))
):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(flow.float)
    return flow.concat([a,b], axis=1)

flow.add_job(ConcatJob)

with flow.Session() as sess:
    a= np.arange(10, dtype=np.float32).reshape(5,2)
    b= np.arange(15, dtype=np.float32).reshape(5,3)
    of_out = sess.run(ConcatJob, a, b).get()

# TensorFlow
tf_out = tf.concat([a,b], 1)

max_diff = np.max(np.abs(of_out - tf_out))
print("Concat max diff:  " + str(max_diff) )
