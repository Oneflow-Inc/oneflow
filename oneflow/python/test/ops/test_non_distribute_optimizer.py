import oneflow as flow
import numpy as np

def test_non_distribute_optimizer(test_case):
    flow.config.gpu_device_num(2)
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.enable_all_reduce_group(True)
    func_config.train.primary_lr(5)
    func_config.train.model_update_conf(dict(naive_conf={}))
    func_config.enable_non_distributed_optimizer(True)
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 10))):
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(100))
        flow.losses.add_loss(x + w)
    Foo(np.ones((2, 10), dtype=np.float32))

def _test_two_job_non_distribute_optimizer(test_case):
    flow.config.gpu_device_num(2)
    flow.config.enable_debug_mode(True)
    eval_config = flow.FunctionConfig()
    eval_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    @flow.function(eval_config)
    def Bar():
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(100))
        return w

    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.enable_all_reduce_group(True)
    func_config.train.primary_lr(5)
    func_config.train.model_update_conf(dict(naive_conf={}))
    func_config.enable_non_distributed_optimizer(True)
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 10))):
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(100))
        flow.losses.add_loss(x + w)
    Foo(np.ones((2, 10), dtype=np.float32))

def _test_non_distribute_optimizer_var_as_loss(test_case):
    flow.config.gpu_device_num(2)
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.enable_all_reduce_group(True)
    func_config.train.primary_lr(5)
    func_config.train.model_update_conf(dict(naive_conf={}))
    func_config.enable_non_distributed_optimizer(True)
    @flow.function(func_config)
    def Foo():
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(100))
        flow.losses.add_loss(w)
    Foo()
