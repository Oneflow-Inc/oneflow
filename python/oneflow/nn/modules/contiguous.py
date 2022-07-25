"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
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
