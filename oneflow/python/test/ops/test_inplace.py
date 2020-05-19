import oneflow as flow
import numpy as np

def MakeFuncConfig(enable_inplace):
    func_config = flow.FunctionConfig()
    func_config.enable_inplace(enable_inplace)
    return func_config

def test_loss_inplace(test_case):
    def IdentityLoss(name):
        w = flow.get_variable(name, (10,), initializer=flow.constant_initializer(100))
        y = flow.math.reduce_sum(w)
        flow.losses.add_loss(y)
        return y
    TrainCompare(test_case, IdentityLoss)

def test_inplace_variable(test_case):
    @flow.function(MakeFuncConfig(True))
    def InplaceVariable():
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(1))
        y = flow.math.relu(w)
        return y
    flow.train.CheckPoint().init()
    test_case.assertTrue(np.allclose(InplaceVariable().get().ndarray(), np.ones((10,), np.float32)))

def test_deadlock(test_case):
    @flow.function(MakeFuncConfig(True))
    def Foo(x = flow.FixedTensorDef((10, ))):
        y = flow.math.relu(x)
        y = flow.math.relu(y)
    Foo(np.ones((10,), dtype=np.float32))

def test_nodeadlock_with_return(test_case):
    @flow.function(MakeFuncConfig(True))
    def Foo(x = flow.FixedTensorDef((10, ))):
        y = flow.math.relu(x)
        y = flow.math.relu(y)
        return y
    Foo(np.ones((10,), dtype=np.float32)).get()

def test_reentrant_lock_check_failed(test_case):
    @flow.function(MakeFuncConfig(True))
    def Foo(x = flow.FixedTensorDef((10, ))):
        y = flow.math.relu(x)
        y = flow.math.relu(y)
    Foo(np.ones((10,), dtype=np.float32))

def test_const_inplace_variable(test_case):
    @flow.function(MakeFuncConfig(True))
    def InplaceVariable():
        w = flow.get_variable("w", (2, 5), initializer=flow.constant_initializer(1))
        y = flow.reshape(w, (10,))
        return y
    flow.train.CheckPoint().init()
    of_ret = InplaceVariable().get().ndarray()
    test_case.assertTrue(np.allclose(of_ret, np.ones((10,), np.float32)))

def TrainCompare(test_case, func):
    lr = 5
    func_config = MakeFuncConfig(True)
    func_config.train.primary_lr(5)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def EnableInplace(): return func('w0')
    
    func_config.enable_inplace(False)
    @flow.function(func_config)
    def DisableInplace(): return func('w1')
    
    flow.train.CheckPoint().init()
    num_iter = 10
    enable_inplace_losses = np.array([EnableInplace().get().tolist() for _ in range(num_iter)])
    disable_inplace_losses = np.array([DisableInplace().get().tolist() for _ in range(num_iter)])
    test_case.assertTrue(np.allclose(enable_inplace_losses, disable_inplace_losses))
