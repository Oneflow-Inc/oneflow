import inspect
import functools

import torch as torch_original
import oneflow.experimental as flow
import numpy as np
from .generators import generator, random_tensor


postulate = ['.rand', '.Tensor']


def torch_tensor_to_flow(x):
    return flow.tensor(x.cpu().numpy())


class PyTorchDoesNotSupportError(Exception):
    pass


def get_args(callable, *args, **kwargs):
    try:
        spec = inspect.getfullargspec(callable)
        for i, arg in enumerate(args):
            arg_name = spec.args[i]
            annotation = spec.annotations[arg_name]
            if isinstance(arg, generator):
                arg.to(annotation)
        for arg_name, arg in kwargs.items():
            annotation = spec.annotations[arg_name]
            if isinstance(arg, generator):
                arg.to(annotation)
    except:
        pass
    pytorch_args, pytorch_kwargs, oneflow_args, oneflow_kwargs = [], {}, [], {}
    def get_pytorch_value(x):
        if isinstance(x, DualObject):
            return x.pytorch
        if isinstance(x, generator):
            return x.value()
        return x

    def get_oneflow_value(x):
        if isinstance(x, DualObject):
            return x.oneflow
        if isinstance(x, generator):
            return x.value()
        return x

    for arg in args:
        
        pytorch_args.append(get_pytorch_value(arg))
        oneflow_args.append(get_oneflow_value(arg))
    for key, value in kwargs.items():
        pytorch_kwargs[key] = get_pytorch_value(value)
        oneflow_kwargs[key] = get_oneflow_value(value)
    return pytorch_args, pytorch_kwargs, oneflow_args, oneflow_kwargs


counter = 0

def GetDualObject(name, pytorch, oneflow):
    global counter
    counter += 1

    skipped_magic_methods = ['__class__', '__mro__', '__new__', '__init__', '__getattr__', '__setattr__', '__getattribute__', '__dict__', '__weakref__', '__builtins__', '__qualname__', '__name__', '__str__', '__repr__']

    pytorch_methods = dir(pytorch)
    if hasattr(pytorch, '__call__') and '__call__' not in pytorch_methods:
        pytorch_methods.append('__call__')

    magic_methods_for_new_cls = {}

    for method_name in pytorch_methods:
        if method_name.startswith('__') and method_name not in skipped_magic_methods:
            # init a new 'method_name' variable other than the one in for loop,
            # avoid a pitfall:
            # https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
            def get_dual_method(method_name):
                # __call__ is special. We should not delegate the '__call__' of the torch wrapper of class 'nn.Conv2d' 
                # to 'nn.Conv2d.__call__', as 'nn.Conv2d.__call__' belongs to the object of type 'nn.Conv2d'
                # (not the class itself)
                if method_name == '__call__':
                    def dual_method(self, *args, **kwargs):
                        pytorch_args, pytorch_kwargs, oneflow_args, oneflow_kwargs = get_args(pytorch, *args, **kwargs)
                        # use () instead of '__call__'
                        try:
                            pytorch_res = pytorch(*pytorch_args, **pytorch_kwargs)
                        except:
                            raise PyTorchDoesNotSupportError()
                        # only check if the method is a postulate when it is called
                        if name in postulate:
                            oneflow_res = torch_tensor_to_flow(pytorch_res)
                        else:
                            oneflow_res = oneflow(*oneflow_args, **oneflow_kwargs)
                        return GetDualObject('unused', pytorch_res, oneflow_res)
                else:
                    def dual_method(self, *args, **kwargs):
                        pytorch_method = getattr(pytorch, method_name)
                        oneflow_method = getattr(oneflow, method_name)
                        pytorch_args, pytorch_kwargs, oneflow_args, oneflow_kwargs = get_args(*args, **kwargs)
                        try:
                            pytorch_res = pytorch_method(*pytorch_args, **pytorch_kwargs)
                        except:
                            raise PyTorchDoesNotSupportError()
                        oneflow_res = oneflow_method(*oneflow_args, **oneflow_kwargs)
                        return GetDualObject('unused', pytorch_res, oneflow_res)
                return dual_method

            magic_methods_for_new_cls[method_name] = get_dual_method(method_name)

    Cls = type(f'{name}_{counter}', (DualObject,), magic_methods_for_new_cls)
    return Cls(name, pytorch, oneflow)


class DualObject:
    def __init__(self, name, pytorch, oneflow):
        self.name = name
        self.pytorch = pytorch
        self.oneflow = oneflow

        if isinstance(pytorch, torch_original.nn.Module):
            state_dict = pytorch.state_dict()
            state_dict = {
                k: v.detach().cpu().numpy() for k, v in state_dict.items()
            }
            oneflow.load_state_dict(state_dict)

    def __str__(self):
        return f'PyTorch object:\n{self.pytorch}\n\nOneFlow object:\n{self.oneflow}'

    def __getattr__(self, key):
        pytorch_attr = getattr(self.pytorch, key)
        oneflow_attr = getattr(self.oneflow, key)
        new_name = f'{self.name}.{key}'

        return GetDualObject(new_name, pytorch_attr, oneflow_attr)


def autotest(n=20):
    def _autotest(f):
        @functools.wraps(f)
        def new_f(test_case):
            nonlocal n
            while n > 0:
                try:
                    res = f(test_case)
                    # TODO: support types other than Tensor, like torch.Size/flow.Size
                    # TODO: support tuple
                    test_case.assertTrue(np.allclose(res.pytorch.cpu().numpy(), res.oneflow.numpy()))
                    n -= 1
                except PyTorchDoesNotSupportError:
                    # TODO: support verbose
                    pass
        return new_f
    return _autotest


def random_pytorch_tensor(ndim=None, dim0=1, dim1=None, dim2=None, dim3=None, dim4=None):
    pytorch_tensor = random_tensor(ndim, dim0, dim1, dim2, dim3, dim4).value()
    flow_tensor = flow.tensor(pytorch_tensor.numpy())
    return GetDualObject('unused', pytorch_tensor, flow_tensor)


torch = GetDualObject('', torch_original, flow)


__all__ = ['torch', 'autotest', 'random_pytorch_tensor']
