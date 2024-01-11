import torch
import oneflow as flow
from oneflow.framework.args_tree import ArgsTree


def input_output_processor(func):
    def process_input(*args, **kwargs):
        def input_fn(value):
            if isinstance(value, torch.Tensor):
                # TODO: https://github.com/siliconflow/sd-team/issues/109
                return flow.utils.tensor.from_torch(value.contiguous())
            else:
                return value

        args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
        out = args_tree.map_leaf(input_fn)
        mapped_args = out[0]
        mapped_kwargs = out[1]
        return mapped_args, mapped_kwargs

    def process_output(output):
        def output_fn(value):
            if isinstance(value, flow.Tensor):
                return flow.utils.tensor.to_torch(value)
            else:
                return value

        out_tree = ArgsTree((output, None), False)
        out = out_tree.map_leaf(output_fn)
        return out[0]

    def wrapper(cls, *args, **kwargs):
        mapped_args, mapped_kwargs = process_input(*args, **kwargs)
        output = func(cls, *mapped_args, **mapped_kwargs)
        return process_output(output)

    return wrapper
