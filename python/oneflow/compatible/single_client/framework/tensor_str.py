import numpy as np
from oneflow.compatible import single_client as flow

def _add_suffixes(tensor_str, suffixes, indent):
    tensor_strs = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind('\n') + 1
    linewidth = 80
    for suffix in suffixes:
        suffix_len = len(suffix)
        if last_line_len + suffix_len + 2 > linewidth:
            tensor_strs.append(',\n' + ' ' * indent + suffix)
            last_line_len = indent + suffix_len
        else:
            tensor_strs.append(', ' + suffix)
            last_line_len += suffix_len + 2
    tensor_strs.append(')')
    return ''.join(tensor_strs)

def _gen_tensor_str(tensor):
    prefix = 'tensor('
    indent = len(prefix)
    suffixes = []
    if tensor.device.type != 'cpu' or (tensor.device.type == 'cuda' and tensor.device.index != 0):
        suffixes.append("device='" + str(tensor.device) + "'")
    suffixes.append('dtype=' + str(tensor.dtype))
    if tensor.grad_fn is not None:
        name = tensor.grad_fn.name()
        suffixes.append('grad_fn=<{}>'.format(name))
    elif tensor.requires_grad:
        suffixes.append('requires_grad=True')
    tensor_str = np.array2string(tensor.numpy(), precision=4, separator=', ', prefix=prefix)
    return _add_suffixes(prefix + tensor_str, suffixes, indent)