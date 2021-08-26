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
from oneflow.ops.builtin_ops import BuiltinOp as builtin_op
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple


def allreduce_fn(ddp_state_for_reversed_params, param):
    def allreduce(grad):
        ddp_state_for_reversed_params[param][0] = True
        ret = None
        for cur_param, (ready, deleted) in ddp_state_for_reversed_params.items():
            if deleted:
                continue
            if ready:
                ddp_state_for_reversed_params[cur_param][1] = True
                if cur_param is param:
                    ret = flow.F.all_reduce(grad)
                else:
                    cur_param.grad = flow.F.all_reduce(cur_param.grad)
            else:
                break
        return ret

    return allreduce


def DistributedDataParallel(
    module: "flow.nn.Module", *, broadcast_buffers: bool = True
):
    world_size = flow.distributed.get_world_size()
    with flow.no_grad():
        for x in module.parameters():
            requires_grad = x.requires_grad
            x.copy_(flow.F.broadcast(x))
            # TODO: fix the bug that x's requires_grad is discarded
            # after flow.F.broadcast
            x.requires_grad_(requires_grad)

    ddp_state_for_reversed_params = OrderedDict(
        reversed([(x, [False, False]) for x in module.parameters()])
    )
    module._ddp_state_for_reversed_params = ddp_state_for_reversed_params
    for param in module.parameters():
        param.register_hook(lambda grad: grad / world_size)
        param.register_hook(allreduce_fn(ddp_state_for_reversed_params, param))

    def post_forward_hook(module, input, output):
        ddp_state_for_reversed_params = module._ddp_state_for_reversed_params
        for state in ddp_state_for_reversed_params.values():
            state[0], state[1] = False, False
        output = flow.F.select_first(
            convert_to_tensor_tuple([output, *ddp_state_for_reversed_params.keys()])
        )
        return output

    module.register_forward_hook(post_forward_hook)

    if broadcast_buffers:

        def pre_forward_hook(module, input):
            with flow.no_grad():
                for x in module.buffers():
                    x.copy_(flow.F.broadcast(x))

        module.register_forward_pre_hook(pre_forward_hook)

    return module
