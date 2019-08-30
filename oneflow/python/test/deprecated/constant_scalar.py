import oneflow as flow

def ConstantScalarJob():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(flow.float)
    dlnet = flow.deprecated.get_cur_job_dlnet_builder()
    return dlnet.ConstantScalar(3.14)
flow.add_job(ConstantScalarJob)

with flow.Session() as sess:
    print(sess.run(ConstantScalarJob).get())
