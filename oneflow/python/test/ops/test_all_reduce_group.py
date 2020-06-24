from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList


def do_test(test_case, mirrored):
    flow.clear_default_session()
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.enable_all_reduce_group(True)
    func_config.train.primary_lr(5)
    func_config.train.model_update_conf(dict(naive_conf={}))
    if mirrored:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())
    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.global_function(func_config)
    def Foo():
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(1))
        flow.losses.add_loss(w)
        return w

    check_point = flow.train.CheckPoint()
    check_point.init()
    r1 = Foo().get().ndarray()
    test_case.assertTrue(np.all(r1 == 1.0))
    r2 = Foo().get().ndarray()
    test_case.assertTrue(np.all(r2 == 0.5))


def test_variable_as_loss_on_two_device(test_case):
    arg_dict = OrderedDict()
    arg_dict["mirrored"] = [True, False]
    for arg in GenArgList(arg_dict):
        do_test(test_case, *arg)
