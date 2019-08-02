import oneflow as flow
import numpy as np

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
#config.grpc_use_no_signal()
flow.init(config)

def DemoJob(x = flow.input_blob_def((10,))):
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    return x
flow.add_job(DemoJob)

with flow.Session() as sess:
    data = []
    for i in range(5): data.append(np.ones((10,), dtype=np.float32) * i)
    print "sess.run(...).get()"
    for x in data: print sess.run(DemoJob, x).get()
    print "sess.map(...).get()"
    for x in sess.map(DemoJob, data).get(): print x
    print "sess.run(...).async_get(...)"
    def PrintRunAsyncResult(x):
        print x
    for x in data: sess.run(DemoJob, x).async_get(PrintRunAsyncResult)
    sess.sync()
    print "sess.map(...).async_get(...)"
    def PrintMapAsyncResult(ndarrays):
        for x in ndarrays: print x
    sess.map(DemoJob, data).async_get(PrintMapAsyncResult)

    # box = flow.Box()
    # sess.map(DemoJob, data).async_get(box.value_setter)
    # import time
    # time.sleep(3)
    # for x in box.value: print x
