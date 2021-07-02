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
from collections import OrderedDict

import oneflow as flow
from oneflow.python.ops.builtin_ops import BuiltinOp as builtin_op
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module


module2params = {}


def allreducefn(reversed_param_list, param, nccl_allreduce_op):
    def allreduce(grad):
        reversed_param_list[param][0] = True
        ret = None
        for cur_param, (ready, deleted, _) in reversed_param_list.items():
            if deleted:
                continue
            if ready:
                reversed_param_list[cur_param][1] = True
                if cur_param == param:
                    ret = nccl_allreduce_op(grad)[0]
                else:
                    cur_param.grad = nccl_allreduce_op(cur_param.grad)[0]
            else:
                break
        return ret

    return allreduce


@oneflow_export("ddp")
def DDP(module: Module, world_size=None):
    if world_size is None:
        world_size = flow.world_size()
    nccl_allreduce_op = (
        builtin_op("eager_nccl_all_reduce")
        .Input("in")
        .Output("out")
        .Attr("parallel_conf", f'device_tag: "gpu", device_name: "0:0-{world_size-1}"',)
        .Build()
    )
    reversed_param_list = OrderedDict(
        reversed([(x, [False, False, name]) for name, x in module.named_parameters()])
    )
    module._reversed_param_list = reversed_param_list
    for _, param in module.named_parameters():
        param.register_hook(allreducefn(reversed_param_list, param, nccl_allreduce_op))

    def hook(module, input, output):
        reversed_param_list = module._reversed_param_list
        for item in reversed_param_list.values():
            item[0], item[1] = False, False
        output = flow.experimental.return_first_input(
            output, *reversed_param_list.keys()
        )
        return output

    module.register_forward_hook(hook)
    return module
