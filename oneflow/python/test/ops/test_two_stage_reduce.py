

import numpy as np
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type

import test_global_storage

def _compare_with_numpy(test_case, np_func, x, y, axis, keepdims=True):
    x = test_global_storage.Get("x")
    dx = test_global_storage.Get("x_diff")
    np_y = np_func(x, axis=axis, keepdims=True)
    test_case.assertTrue(np.allclose(y, np_y, rtol=1e-5, atol=1e-5))  
    mask = np.where(x==y, 1, 0)
    count = np.add.reduce(mask, axis=axis, keepdims=True)
    np_dx = np.where(x==y, 1/count, 0)
    test_case.assertTrue(np.allclose(dx, np_dx, rtol=1e-5, atol=1e-5)) 

def _test_two_stage_reduce(test_case, flow_func, np_func, device_type, axis):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))
    func_config.use_boxing_v2(True)


    @flow.function(func_config)
    def two_stage_reduce_job(x=flow.FixedTensorDef((4, 10, 30, 20))):
        with flow.fixed_placement(device_type, '0:0'):
            x += flow.get_variable(name='v1', shape=(1,),
                                   dtype=flow.float, initializer=flow.zeros_initializer())
        with flow.fixed_placement(device_type, '0:0-3'):
            loss = flow_func(x.with_distribute(flow.distribute.split(1)), axis=axis, keepdims=True)
            loss = flow.identity(loss)
            flow.losses.add_loss(loss)
            
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            return loss
        

    x = np.random.randint(low=0, high=10, size=(4, 10, 30, 20)).astype(np.float32)
    y = two_stage_reduce_job(x).get().ndarray()
    _compare_with_numpy(test_case, np_func, x, y, axis=tuple(axis))


def test_two_stage_reduce_max(test_case):
    arg_dict = OrderedDict()
    arg_dict['flow_func'] = [flow.math.two_stage_reduce_max]
    arg_dict['np_func'] = [np.maximum.reduce]
    arg_dict['device_type'] = ['cpu', 'gpu']
    arg_dict['axis'] = [[1], [1, 2], [1, 2, 3]]

    for arg in GenArgList(arg_dict):
        _test_two_stage_reduce(test_case, *arg)

def test_two_stage_reduce_min(test_case):
    arg_dict = OrderedDict()
    arg_dict['flow_func'] = [flow.math.two_stage_reduce_min]
    arg_dict['np_func'] = [np.minimum.reduce]
    arg_dict['device_type'] = ['cpu', 'gpu']
    arg_dict['axis'] = [[1], [1, 2], [1, 2, 3]]

    for arg in GenArgList(arg_dict):
        _test_two_stage_reduce(test_case, *arg)
