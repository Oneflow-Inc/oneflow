import oneflow as flow


def ContiguousGrad(module):

    def grad_setting_fn(module, param):
        def grad_setting(grad):
            if param.grad is None:
                tmp = module._grad_buf[param]
                # print('tensor info:')
                # print(flow._oneflow_internal.dtr.tensor_info(tmp))
                param.grad = tmp
                param._is_grad_acc_inplace = True
            return grad

        return grad_setting

    module._grad_buf = {p: flow.zeros_like(p) for p in module.parameters()}
    for grad in module._grad_buf.values():
        flow._oneflow_internal.dtr.set_non_evictable(grad)

    for param in module.parameters():
        param.register_hook(grad_setting_fn(module, param))
