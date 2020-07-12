import numpy as np
import oneflow as flow


def test_dynamic_reshape(test_case):
    num_gpus = 2
    data_shape = (10, 10, 10)
    flow.config.gpu_device_num(num_gpus)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def DynamicReshapeJob(x=flow.MirroredTensorDef(data_shape)):
        reshape_out1 = flow.reshape(x, (-1, 20))
        my_model = flow.get_variable(
            "my_model",
            shape=(20, 32),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
            trainable=True,
        )
        mm_out = flow.matmul(reshape_out1, my_model)
        reshape_out2 = flow.reshape(mm_out, (-1, 8, 4))
        flow.losses.add_loss(reshape_out2)
        return reshape_out1

    data = [np.random.rand(*data_shape).astype(np.float32) for i in range(num_gpus)]
    out = DynamicReshapeJob(data).get().ndarray_list()
    for i in range(num_gpus):
        test_case.assertTrue(np.array_equal(np.reshape(data[i], (50, 20)), out[i]))
