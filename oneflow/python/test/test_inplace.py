import oneflow as flow
import numpy as np

def IdentityLoss(name):
    w = flow.get_variable(name, (10,), initializer=flow.constant_initializer(100))
    y = flow.math.reduce_sum(w)
    flow.losses.add_loss(y)
    return y

def test_loss_inplace(test_case):
    Compare(test_case, IdentityLoss)

def Compare(test_case, func):
    lr = 5
    @flow.function
    def EnableInplace():
        flow.config.enable_inplace(True)
        flow.config.train.primary_lr(lr)
        flow.config.train.model_update_conf(dict(naive_conf={}))
        return IdentityLoss('w0')
    
    @flow.function
    def DisableInplace():
        flow.config.enable_inplace(False)
        flow.config.train.primary_lr(lr)
        flow.config.train.model_update_conf(dict(naive_conf={}))
        return IdentityLoss('w1')
    
    flow.train.CheckPoint().init()
    num_iter = 10
    enable_inplace_losses = np.array([EnableInplace().get() for _ in range(num_iter)])
    disable_inplace_losses = np.array([DisableInplace().get() for _ in range(num_iter)])
    print(enable_inplace_losses)
    print(disable_inplace_losses)
    test_case.assertTrue(np.allclose(enable_inplace_losses, disable_inplace_losses))
