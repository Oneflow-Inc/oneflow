import oneflow as flow
import numpy as np


config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
# config.ctrl_port(12311)
# config.grpc_use_no_signal()
flow.init(config)

def ReluJob(x = flow.input_blob_def(shape = (10,), is_dynamic=True)):
    return flow.keras.activations.relu(x)
flow.add_job(ReluJob)

def TestResult(a, b):
    result = np.isclose(a, b, rtol=1e-03, atol=1e-05)
    for i in result:
        assert i, "the test is wrong!"

with flow.Session() as sess:
    TestResult(sess.run(ReluJob, np.array(range(10), dtype=np.float32)).get(), np.array(range(10), dtype=np.float32))
    TestResult(sess.run(ReluJob, np.array([-2, 5, -5, 2], dtype=np.float32)).get(), np.array([0, 5, 0, 2], dtype=np.float32))
    TestResult(sess.run(ReluJob, np.array([2, 0, 0, 5, -7, 9], dtype=np.float32)).get(), np.array([2, 0, 0, 5, 0, 9], dtype=np.float32))
    TestResult(sess.run(ReluJob, np.array([1], dtype=np.float32)).get(), np.array([1], dtype=np.float32))


