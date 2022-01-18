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


def grad_setting_fn(module, param):
    def grad_setting(grad):
        if param.grad is None:
            start = module._param_grad_start[param]
            bucket_index = module._bucket_index[param]
            # print('setting .grad: bucket_index={}'.format(bucket_index))
            bucket_tensor = module._bucket_tensors[bucket_index]
            param.grad = bucket_tensor[start : start + param.numel()].view(param.shape)
        return grad
    return grad_setting


def allreduce_fn(module, param):
    ddp_state_for_reversed_params = module._ddp_state_for_reversed_params
    buckets = module._buckets
    bucket_tensors = module._bucket_tensors
    def allreduce(grad):
        # import pdb; pdb.set_trace()
        # print('hook')
        ddp_state_for_reversed_params[param][0] = True
        for index, bucket in enumerate(buckets):
            # print('bucket:')
            # print(bucket)
            deleted = all(ddp_state_for_reversed_params[x][1] for x in bucket)
            if deleted:
                continue

            assert not any(ddp_state_for_reversed_params[x][1] for x in bucket)

            all_params_in_bucket_ready = all(
                ddp_state_for_reversed_params[x][0] for x in bucket
            )
            # print([ddp_state_for_reversed_params[x][0] for x in bucket])
            if all_params_in_bucket_ready:
                for x in bucket:
                    ddp_state_for_reversed_params[x][1] = True
                # print(f'all_grad={module._all_grad}')
                # print(f'allreduce {index}')
                # print(f'bucket_tensors[{index}]: {bucket_tensors[index]}')
                # for x in bucket:
                    # print(f'x.grad={x.grad}')
                    # print(f'grad={grad}')
                flow._C.local_all_reduce(bucket_tensors[index], inplace=True)
                # print(f'bucket_tensors[{index}]: {bucket_tensors[index]}')
            else:
                break

    return allreduce


def DistributedDataParallel(
    module: "flow.nn.Module", *, broadcast_buffers: bool = True, bucket_size: int = 10
):
    # print(f'bucket_size={bucket_size}')
    assert all(x.dtype == flow.float32 for x in module.parameters())

    world_size = flow.env.get_world_size()
    with flow.no_grad():
        for x in module.parameters():
            requires_grad = x.requires_grad
            flow._C.broadcast(x, inplace=True)
            # TODO: fix the bug that x's requires_grad is discarded
            # after flow._C.broadcast
            x.requires_grad_(requires_grad)

    # module._bucket_indexes = {}
    all_grad_size = sum([x.numel() for x in module.parameters()])
    if all_grad_size > 0:
        device = list(module.parameters())[0].device
        assert all(x.device == device for x in module.parameters())
    reversed_param_list = list(reversed(list(module.parameters())))
    module._param_grad_start = {}
    bytes = 0
    with flow.no_grad():
        for i, x in enumerate(reversed_param_list):
            assert x.is_leaf
            if i % bucket_size == 0:
                bytes = 0
            module._param_grad_start[x] = bytes
            # x.grad = module._all_grad[bytes : bytes + x.numel()]
            bytes += x.numel()

    module._bucket_index = {x: i // bucket_size for i, x in enumerate(reversed_param_list)}
    # print(f'bucket_index={module._bucket_index}')
    module._buckets = [reversed_param_list[i : i + bucket_size] for i in range(0, len(reversed_param_list), bucket_size)]
    # print(f'len of buckets: {len(module._buckets)}')
    # print(f'buckets: {module._buckets}')
    # print(f'reversed_param_list: {reversed_param_list}')
    # assert bytes == all_grad_size

    # bucket_start_byte = 0
    bucket_bytes = 0
    module._bucket_tensors = []
    for b in module._buckets:
        bucket_bytes = sum([x.numel() for x in b])
        module._bucket_tensors.append(flow.zeros(bucket_bytes, dtype=flow.float32, device=device))
        # bucket_start_byte += bucket_bytes
    # assert bucket_start_byte == all_grad_size

    ddp_state_for_reversed_params = OrderedDict(
        reversed([(x, [False, False]) for x in module.parameters() if x.requires_grad])
    )
    module._ddp_state_for_reversed_params = ddp_state_for_reversed_params
    for param in module.parameters():
        param.register_hook(lambda grad: grad / world_size)
        param.register_hook(grad_setting_fn(module, param))
        param._register_post_hook(allreduce_fn(module, param))

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
                for x in module.buffers():
                    flow._C.broadcast(x, inplace=True)

        module.register_forward_pre_hook(pre_forward_hook)

    return module

