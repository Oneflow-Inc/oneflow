import oneflow as flow
import numpy as np
def test_variable_as_loss_on_two_device(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.enable_all_reduce_group(True)
    func_config.train.primary_lr(5)
    func_config.train.model_update_conf(dict(naive_conf={}))
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    @flow.function(func_config)
    def Foo():
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(0))
        flow.losses.add_loss(w)
        return w
    check_point = flow.train.CheckPoint()
    check_point.init()
    r1 = Foo().get().ndarray()
    r2 = Foo().get().ndarray()
    assert np.all(r2 == -0.5)
