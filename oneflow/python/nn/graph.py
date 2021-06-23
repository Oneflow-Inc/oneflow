from __future__ import absolute_import
from collections import OrderedDict, namedtuple
from typing import Union, TypeVar, Iterator, Optional, Set, Tuple, Dict, List, Callable

import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import Tensor
from oneflow.python.nn.parameter import Parameter
from oneflow.python.nn.optimizer.optimizer import Optimizer
from oneflow.python.framework.function_util import FunctionConfig

@oneflow_export("experimental.nn.Graph")
class Graph(object):
    def __init__(self):
        self.training = True
        self.config = GraphConfig()
        self._nodes = OrderedDict()
        self._optimizers = OrderedDict()
        self._runnable_func = None

    def build(self, *args):
        raise NotImplementedError()

    def __call__(self, *args):  
        if self._runnable_func is None:
            # TODO(): implement compile
            self._runnable_func = vm_api.compile_job(self, *args)
        return self._runnable_func(*args)

    def add_module(self, name: str, module: Module = None) -> None:
        r"""Adds a child module to the current graph.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this graph using the given name
            module (Module): child module to be added to the graph.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(type(name)))
        elif hasattr(self, name) and name not in self._nodes:
            raise KeyError("attribute '{}' already exists".format(name))
        elif "." in name:
            raise KeyError('module name can\'t contain ".", got: {}'.format(name))
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        self._nodes[name] = Node(name, module)

    def add_optimizer(
        self,
        name: str,
        optimizer: Optimizer = None,
        lr_scheduler = None, 
        grad_clipping_conf = None,
        weight_decay_conf = None,
    ):
        self._optimizers[name] = self.OptimizerConfig(optimizer, lr_scheduler, grad_clipping_conf, weight_decay_conf)
    
    def train(self, mode: bool = True):
        self.training = mode

    def __setattr__(self, name: str, value = None):
        if isinstance(value, Module):
            print("graph add module attr: ", name)
            self.add_module(name, value)
        elif isinstance(value, Optimizer):
            raise AttributeError(
                "'{}' object are not allowed to set Optimizer attribute named '{}', please use add_optimizer(...) instead.".format(type(self).__name__, name)
            )
        else:
            print("graph add other type attr: ", name)
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        if "_nodes" in self.__dict__:
            if name in self._nodes:
                return self._nodes[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )


class Node(object):
    def __init__(self, name: str, value: Union[Module, Parameter, Tensor] = None):
        print(">>>", name, " node start creating")
        assert not isinstance(value, Node)
        self._name = name
        self._type = ""
        self._origin = value
        self._config = NodeConfig() 
    
        if isinstance(value, Module):
            print("create module node: ", name)
            self._type = "module"
            self._modules = OrderedDict()
            self._parameters = OrderedDict()
            self._buffers = OrderedDict()
            for n, m in list(value.named_children()):
                print("node ", name, " has sub module n ", n, " module ", type(m))
                self.__setattr__(n, Node(n, m))
            # for n, p in list(value.named_parameters()):
            #     print("node ", name, " has parameter n", n, " p ", type(p))
            #     # self.__setattr__(n, Node(n, p))
            # for n, b in list(value.named_buffers()):
            #     print("node ", name, " has buffer n", n, " b ", type(b))
            #     # self.__setattr__(n, Node(n, b))
        elif isinstance(value, Parameter):
            print("create parameter node: ", name)
            self._type = "parameter"
        elif isinstance(value, Tensor):
            print("create buffer node: ", name)
            self._type = "buffer"
        else:
            raise NotImplementedError()
        print("<<<", name, " node created.")

    def __call__(self, *args):
        if self._type == "module":
            return self._origin.__class__.__call__(self, *args) 
        # TODO(): deal with parameter and buff
    
    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type
    
    @property
    def origin(self):
        return self._origin
    
    def __setattr__(self, name: str, value = None) -> None:
        if value is None or not isinstance(value, Node):
            self.__dict__[name] = value
        else:
            dicts_or_sets = (self.__dict__, self._modules, self._parameters, self._buffers)
            for d in dicts_or_sets:
                if name in d:
                    raise AttributeError(
                        "'{}' object has duplicated attribute named '{}'".format(self._name, name)
                    )
            if value.type == "module":
                self._modules[name] = value
            elif value.type == "parameter":
                self._parameters[name] = value 
            elif value.type == "buffer":
                self._buffers[name] = value
            else:
                raise AttributeError(
                    "'{}' object are not allowed to set attribute named '{}'".format(type(self).__name__, name)
                )

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self.__dict__[name]
        
        if self._type == "module":
            if "_modules" in self.__dict__:
                modules = self.__dict__["_modules"]
                if name in modules:
                    return modules[name]
            if "_parameters" in self.__dict__:
                _parameters = self.__dict__["_parameters"]
                if name in _parameters:
                    # TODO(): return node when need config
                    return _parameters[name].origin
            if "_buffers" in self.__dict__:
                _buffers = self.__dict__["_buffers"]
                if name in _buffers:
                    # TODO(): return node when need config
                    return _buffers[name].origin
            if name in self._origin.__dict__:
                return self._origin.__dict__[name]

        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

class GraphConfig(FunctionConfig):
    def __init__(self):
        super().__init__()

class NodeConfig(object):
    def __init__(self):
        # TODO(): implement config for node
        pass

class OptimizerConfig(object):
    def __init__(
        self,
        name: str,
        optimizer: Optimizer = None,
        lr_scheduler = None, 
        grad_clipping_conf = None,
        weight_decay_conf = None,
    ):
        self.name = name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_clipping_conf = grad_clipping_conf
        self.weight_decay_conf = weight_decay_conf

