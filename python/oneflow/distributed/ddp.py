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
import oneflow.nn as nn
from oneflow.ops.builtin_ops import BuiltinOp as builtin_op
from oneflow.nn.module import Module
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple


def allreducefn(reversed_param_list, param, allreduce_module):
    def allreduce(grad):
        reversed_param_list[param][0] = True
        ret = None
        for cur_param, (ready, deleted, _) in reversed_param_list.items():
            if deleted:
                continue
            if ready:
                reversed_param_list[cur_param][1] = True
                if cur_param == param:
                    ret = allreduce_module(grad)[0]
                else:
                    cur_param.grad = allreduce_module(cur_param.grad)[0]
            else:
                break
        return ret

    return allreduce


def ddp(module: Module):
    world_size = flow.framework.distribute.get_world_size()
    allreduce_module = nn.AllReduce(list(range(world_size)))
    reversed_param_list = OrderedDict(
        reversed([(x, [False, False, name]) for name, x in module.named_parameters()])
    )
    module._reversed_param_list = reversed_param_list
    for _, param in module.named_parameters():
        param.register_hook(allreducefn(reversed_param_list, param, allreduce_module))

    def hook(module, input, output):
        reversed_param_list = module._reversed_param_list
        for item in reversed_param_list.values():
            item[0], item[1] = False, False
        output = flow.F.return_first_input(
            convert_to_tensor_tuple([output, *reversed_param_list.keys()])
        )
        return output

    module.register_forward_hook(hook)
    return module
