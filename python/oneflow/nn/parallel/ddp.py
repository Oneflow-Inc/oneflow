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
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple


def allreduce_fn(ddp_state_for_reversed_params, param):
    def allreduce(grad):
        ddp_state_for_reversed_params[param][0] = True
        for cur_param, (ready, deleted) in ddp_state_for_reversed_params.items():
            if deleted:
                continue
            if ready:
                ddp_state_for_reversed_params[cur_param][1] = True
                # NOTE(jianhao)(higher-order-grad): local allreduce doesn't have gradient function, higher-order grad may be unsupported
                if cur_param is param:
                    flow._C.local_all_reduce(grad, True)
                else:
                    flow._C.local_all_reduce(cur_param.grad, True)
            else:
                break

    return allreduce


def DistributedDataParallel(
    module: "flow.nn.Module", *, broadcast_buffers: bool = True
):
    world_size = flow.env.get_world_size()
    with flow.no_grad():
        for x in module.parameters():
            requires_grad = x.requires_grad
            flow._C.broadcast(x, inplace=True)
            # TODO: fix the bug that x's requires_grad is discarded
            # after flow._C.broadcast
            x.requires_grad_(requires_grad)

    ddp_state_for_reversed_params = OrderedDict(
        reversed([(x, [False, False]) for x in module.parameters() if x.requires_grad])
    )
    module._ddp_state_for_reversed_params = ddp_state_for_reversed_params
    # The gradient shoule be averaged by all the nodes, so besides allreduce,
    # a division by world_size is required.
    # Use x * (1 / world_size) instead of x / world_size for two reasons:
    # 1. multiplication is faster than division
    # 2. An inplace operation is needed here (for allreduce grouping)
    #    But we do not have inplace division in oneflow.
    mul_factor = 1 / world_size

    def inplace_mul_and_return_none(x):
        x.mul_(mul_factor)
        return None

    for param in module.parameters():
        param._register_post_grad_accumulation_hook(inplace_mul_and_return_none)
        param._register_post_grad_accumulation_hook(
            allreduce_fn(ddp_state_for_reversed_params, param)
        )

    def post_forward_hook(module, input, output):
        ddp_state_for_reversed_params = module._ddp_state_for_reversed_params
        for state in ddp_state_for_reversed_params.values():
            state[0], state[1] = False, False
        if isinstance(output, (tuple, list)):
            if isinstance(output[0], dict):
                # For List[Dict[Tensor]] return type.
                out_key_list = []
                out_val_list = []
                for out in output:
                    out_keys = list(out.keys())
                    out_values = list(out.values())
                    out_key_list.append(out_keys)
                    out_val_list.extend(out_values)
                out_values = flow._C.select_top_n(
                    convert_to_tensor_tuple(
                        [*out_val_list, *ddp_state_for_reversed_params.keys()]
                    ),
                    n=len(out_val_list),
                )
                output = []
                for i, keys in enumerate(out_key_list):
                    output.append(
                        dict(zip(keys, out_values[i * len(keys) : (i + 1) * len(keys)]))
                    )
                return output
            else:
                # For List[Tensor] return type.
                output = flow._C.select_top_n(
                    convert_to_tensor_tuple(
                        [*output, *ddp_state_for_reversed_params.keys()]
                    ),
                    n=len(output),
                )
        elif isinstance(output, dict):
            # For Dict[Tensor] return type.
            out_keys = list(output.keys())
            out_values = list(output.values())
            out_values = flow._C.select_top_n(
                convert_to_tensor_tuple(
                    [*out_values, *ddp_state_for_reversed_params.keys()]
                ),
                n=len(out_values),
            )
            return dict(zip(out_keys, out_values))
        else:
            # For Tensor return type.
            output = flow._C.select_top_n(
                convert_to_tensor_tuple(
                    [output, *ddp_state_for_reversed_params.keys()]
                ),
                n=1,
            )[0]
        return output

    module.register_forward_hook(post_forward_hook)

    if broadcast_buffers:

        def pre_forward_hook(module, input):
            with flow.no_grad():
                buffers = list(module.buffers())
                if len(buffers) > 0:
                    flow._C.stream_touch(buffers)  # for reusing soft syncs
                for x in buffers:
                    flow._C.broadcast(x, inplace=True)

        module.register_forward_pre_hook(pre_forward_hook)

    return module
