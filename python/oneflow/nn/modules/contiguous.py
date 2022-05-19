import oneflow as flow


def ContiguousGrad(module):

    def grad_setting_fn(module, param):
        def grad_setting(grad):
            assert param.grad is None
            param.grad = module._grad_buf[param]
            param._is_grad_acc_inplace = True
            return grad

        return grad_setting

    module._grad_buf = {p: flow.zeros_like(p) for p in module.parameters()}

    for param in module.parameters():
        param.register_hook(grad_setting_fn(module, param))
