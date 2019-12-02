import oneflow as flow
import numpy as np

def test_loss_inplace(test_case):
    def IdentityLoss(name):
        w = flow.get_variable(name, (10,), initializer=flow.constant_initializer(100))
        y = flow.math.reduce_sum(w)
        flow.losses.add_loss(y)
        return y
    TrainCompare(test_case, IdentityLoss)

def test_inplace_variable(test_case):
    @flow.function
    def InplaceVariable():
        flow.config.enable_inplace(True)
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(1))
        y = flow.math.relu(w)
        return y
    flow.train.CheckPoint().init()
    test_case.assertTrue(np.allclose(InplaceVariable().get(), np.ones((10,), np.float32)))

def test_deadlock(test_case):
    @flow.function
    def Foo(x = flow.input_blob_def((10, ))):
        flow.config.enable_inplace(True)
        y = flow.math.relu(x)
        y = flow.math.relu(y)
    Foo(np.ones((10,), dtype=np.float32))

def test_nodeadlock_with_return(test_case):
    @flow.function
    def Foo(x = flow.input_blob_def((10, ))):
        flow.config.enable_inplace(True)
        y = flow.math.relu(x)
        y = flow.math.relu(y)
        return y
    Foo(np.ones((10,), dtype=np.float32)).get()

def test_reentrant_lock_check_failed(test_case):
    @flow.function
    def Foo(x = flow.input_blob_def((10, ))):
        flow.config.enable_inplace(True)
        y = flow.math.relu(x)
        y = flow.math.relu(y)
    Foo(np.ones((10,), dtype=np.float32))

def test_const_inplace_variable(test_case):
    @flow.function
    def InplaceVariable():
        flow.config.enable_inplace(True)
        w = flow.get_variable("w", (2, 5), initializer=flow.constant_initializer(1))
        y = flow.reshape(w, (10,))
        return y
    flow.train.CheckPoint().init()
    of_ret = InplaceVariable().get()
    print(of_ret)
    test_case.assertTrue(np.allclose(of_ret, np.ones((10,), np.float32)))

def TrainCompare(test_case, func):
    lr = 5
    @flow.function
    def EnableInplace():
        flow.config.enable_inplace(True)
        flow.config.train.primary_lr(lr)
        flow.config.train.model_update_conf(dict(naive_conf={}))
        return func('w0')
    
    @flow.function
    def DisableInplace():
        flow.config.enable_inplace(False)
        flow.config.train.primary_lr(lr)
        flow.config.train.model_update_conf(dict(naive_conf={}))
        return func('w1')
    
    flow.train.CheckPoint().init()
    num_iter = 10
    enable_inplace_losses = np.array([EnableInplace().get() for _ in range(num_iter)])
    disable_inplace_losses = np.array([DisableInplace().get() for _ in range(num_iter)])
    print(enable_inplace_losses)
    print(disable_inplace_losses)
    test_case.assertTrue(np.allclose(enable_inplace_losses, disable_inplace_losses))
