import os
import oneflow as flow
import numpy as np
#from oneflow.python.test.test_util import Save


flow.config.gpu_device_num(1)

of_dtype2np = {
    flow.float: np.float32,
    flow.double: np.float64,
}

of_activation_map = {
    "relu": flow.nn.relu,
    "sigmoid": flow.keras.activations.sigmoid,
    "tanh": flow.keras.activations.tanh,
    "old_relu": flow.keras.activations.relu,
    # "gelu": flow.keras.activations.gelu,
}

def activation_forward_test(device, activation_type, dtype, shape):
    assert device in ['cpu', 'gpu']
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(dtype)
    
    @flow.function(func_config)
    def ForwardOnlyJob(x = flow.FixedTensorDef(shape, dtype=dtype)):
        with flow.device_prior_placement(device, "0:0"):
            return of_activation_map[activation_type](x)

    x = np.random.rand(*shape).astype(of_dtype2np[dtype]) - 0.5
    print(x)
    of_out = ForwardOnlyJob(x).get()
    print(of_out.ndarray())

def GetSavePath():
    return "./log/op_unit_test/"

def Save(name):
    path = GetSavePath()
    if not os.path.isdir(path):
        print('make dir')
        os.makedirs(path)

    def _save(x):
        np.save(os.path.join(path, name), x.ndarray())

    return _save

def activation_test(device, activation_type, dtype, shape):
    assert device in ["gpu", "cpu"]
    flow.clear_default_session()
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def ActivationJob():
        with flow.device_prior_placement(device, "0:0"):
            x = flow.get_variable(
                "x",
                shape=shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = of_activation_map[activation_type](x)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            return loss

    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ActivationJob().get()
    print(of_out.ndarray())

if __name__ == "__main__":
    device_type = 'cpu'
    activate_type = 'relu'
    data_type = flow.float
    shape = (4, 4)
    #activation_forward_test(device_type, activate_type, data_type, shape)
    activation_test(device_type, activate_type, data_type, shape)