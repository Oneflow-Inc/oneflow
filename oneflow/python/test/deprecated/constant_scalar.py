import oneflow as flow

flow.config.gpu_device_num(1)


@flow.global_function()
def ConstantScalarJob():
    return dlnet.ConstantScalar(3.14)


print(ConstantScalarJob().get())
