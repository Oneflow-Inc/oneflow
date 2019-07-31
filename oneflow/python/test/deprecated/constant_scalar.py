import oneflow as flow

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
config.grpc_use_no_signal()
flow.init(config)

jobs = []
@flow.append_func_to_list(jobs)
def ConstantScalarJob():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(flow.float)
    dlnet = flow.deprecated.get_cur_job_dlnet_builder()
    return dlnet.ConstantScalar(3.14)

with flow.Session(jobs) as sess:
    print(sess.run(ConstantScalarJob).get())
