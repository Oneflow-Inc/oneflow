import oneflow as flow
import numpy as np

def test_no_watch_scope_consistent(test_case): 
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float32)
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        return flow.layers.batch_normalization(x)
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))

def TODO_test_no_watch_scope(test_case): 
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        return flow.layers.batch_normalization(x)
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))

def test_train_consistent(test_case): 
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float32)
    func_config.train.primary_lr(0.001)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        y = flow.layers.batch_normalization(x, axis=1)
        flow.losses.add_loss(flow.math.reduce_sum(y))
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))

def TODO_test_train(test_case): 
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.train.primary_lr(0.001)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        y = flow.layers.batch_normalization(x, axis=1)
        flow.losses.add_loss(flow.math.reduce_sum(y))
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))

def test_watch_scope(test_case): 
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float32)
    func_config.train.primary_lr(0.001)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        with flow.watch_scope({}, {}):
            y = flow.layers.batch_normalization(x, axis=1)
        flow.losses.add_loss(flow.math.reduce_sum(y))
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))
