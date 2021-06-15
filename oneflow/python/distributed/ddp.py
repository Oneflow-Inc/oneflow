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
import os
from collections import OrderedDict

import oneflow as flow
from oneflow.python.ops.builtin_ops import BuiltinOp as builtin_op
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module


module2params = {}


def allreducefn(param_list, param, name, num):
    # print(f"original grad: {grad}")
    def allreduce(grad):
        param_list[param] = True
        ret = None
        for cur_param, ready in param_list.items():
            if ready:
                del param_list[cur_param]
                print(f"allreduce: rank: {flow.local_rank()}, name={name}")
                op = (
                    builtin_op("eager_nccl_all_reduce")
                    .Input("in")
                    .Output("out")
                    .Attr(
                        "parallel_conf",
                        f'device_tag: "gpu", device_name: "0:0-{num-1}"',
                    )
                    .Build()
                )
                if cur_param.grad is None:
                    ret = op(grad)[0]
                else:
                    cur_param.grad = op(cur_param.grad)[0]
            else:
                break
        return ret

    return allreduce


def identity(x):
    op = builtin_op("identity").Input("in").Output("out").Build()
    return op(x)[0]


@oneflow_export("ddp")
def DDP(module: Module):
    num = flow.world_size()
    param_list = OrderedDict(reversed([(x, False) for x in module.parameters()]))
    module2params[module] = param_list
    for name, param in module.named_parameters():
        param.register_hook(allreducefn(param_list, param, name, num))

    def hook(module, input, output):
        if flow.local_rank() != 0:
            identity(module.fc2.weight)
        # for param in module.parameters():
        # identity(param)

    # module.register_forward_hook(hook)
    return module
