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
import sys
from collections import OrderedDict

from oneflow.framework.tensor import Tensor
from typing import Callable, Dict, Union, List, Tuple

def add_indent(in_s, num_spaces):
    s = in_s.split("\n")
    if len(s) == 1:
        return in_s
    first = s.pop(0)
    s = [num_spaces * " " + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def sys_exc_error_msg():
    msg = ""
    exc_info = sys.exc_info()
    if len(exc_info) > 0:
        msg += str(exc_info[0])
    if len(exc_info) > 1:
        msg += " " + str(exc_info[1])
    return msg


def seq_to_func_return(seq, need_unpack=False):
    if need_unpack:
        return seq[0]
    return seq

class IOMapper(object):
    def __init__(self, io_values: Union[Tuple, List, Dict], map_function:Callable) -> None:
        assert io_values != None
        assert map_function != None

        self._io_values = io_values
        self._mapped_io_values = None
        self._map_function = map_function
        self._pre_hooks = []
        self._post_hooks = []

    def clear_mapping_result(self):
        self._mapped_io_values = None 
    
    def is_mapped(self):
        self._mapped_io_values != None 

    def register_pre_hook(self, hook:Callable):
        assert hook != None 
        self._pre_hooks.append(hook)

    def remove_last_pre_hook(self):
        assert len(self._pre_hooks) > 0
        self._pre_hooks.pop()

    def remove_last_post_hook(self):
        assert len(self._post_hooks) > 0
        self._post_hooks.pop()

    def register_post_hook(self, hook:Callable):
        assert hook != None 
        self._post_hooks.append(hook)

    def get_mapping_result(self):
        if self.is_mapped():
            return self._mapped_io_values

        def execute_mapping(value):
            for pre_hook in self._pre_hooks:
                pre_hook(value)

            if isinstance(value, tuple) or isinstance(value, list):
                mapped_value = value.__class__(map(lambda x: execute_mapping(x), value))
            elif isinstance(value, dict):
                mapped_value = dict(map(lambda x: (x[0], execute_mapping(x[1])), value.items()))
            else:
                mapped_value = self._map_function(value)

            for post_hook in self._post_hooks:
                post_hook(mapped_value)

            return mapped_value

        self._mapped_io_values = execute_mapping(self._io_values)
        return self._mapped_io_values

    def get_flattened(self):
        flattened = []
        def flatten_hook(value):
            if isinstance(value, Tensor):
                flattened.append(value)
        
        self.register_post_hook(flatten_hook)
        self.clear_mapping_result()
        self.get_mapping_result()
        self.remove_last_post_hook()
        return flattened

class NamedIOMapper(IOMapper):
    class IONodeType:
        TENSOR = "TENSOR"
        NONE = "NONE"
        LIST = "LIST"
        TUPLE = "TUPLE"
        DICT = "DICT"
        OPAQUE = "OPAQUE"

    class IONode(object):
        def __init__(self, name = None, start_idx = 0, value = None, prefix = "") -> None:
            self._name = name if name is not None else str(start_idx)
            self._prefix = prefix
            self._start_idx = start_idx
            self._end_idx = start_idx
            self._cur_level_idx = -1
            self._child_io_nodes = OrderedDict()
            self._value = value 

        def size(self):
            return self._end_idx - self._start_idx + 1 

        def add_child_io_node(self, node):
            self._child_io_nodes[self._cur_level_idx + 1] = node 
            self._end_idx += node.size()
            self._cur_level_idx += 1 

        def named_nodes(self, memo=None):
            if memo is None:
                momo = set()
            if self not in memo:
                memo.add(self)
                yield (self._prefix + "_" + str(self._name), self)
                for (level_idx, node) in self._child_io_nodes.items():
                    if node is None: 
                        continue
                    for n in node.named_nodes(memo):
                        yield n 

        def __repr__(self):
            repr_str = ""
            repr_str += "(name: " + self._name
            repr_str += ", idx: " + str(self._start_idx)
            repr_str += ", type: " + self._type
            if self._type == IONodeType.TENSOR:
                repr_str += ", value: " + self._value._meta_repr() + ")"
            else:
                repr_str += ", value: " + repr(self._value) + ")"
            return repr_str

    def __init__(self, root_prefix:str, root_name:str, io_values: Union[Tuple, List, Dict], map_function: Callable) -> None:
        super().__init__(io_values, map_function)
        self._root_prefix = root_prefix 
        self._root_name = root_name 
        self._root_io_node = None 

    def get_pre_mapping_named_io_values(self):
        # we build an IONode tree first

        current_io_node = None 
        last_io_node = None 
        def pre_hook(value):
            if self._root_io_node == None:
                self._root_io_node = self.IONode(self._root_name, 0, value, self._root_prefix) 
                current_io_node = self._root_io_node 
            else:
                last_io_node = current_io_node 
                child_node_name = None 
                child_node_prefix = 
                if is
                current_io_node.add_child_io_node(self.IONode())
        
        def post_hook(value):
            pass 

        self.clear_mapping_result()
        self.register_pre_hook(pre_hook)
        self.get_mapping_result() 
        self.remove_last_pre_hook()

        return self._root_io_node.named_nodes()

    def get_post_mapping_named_io_values(self):
        pass

# class IONodeType:
#     TENSOR = "TENSOR"
#     NONE = "NONE"
#     LIST = "LIST"
#     TUPLE = "TUPLE"
#     DICT = "DICT"
#     OPAQUE = "OPAQUE"


# class IONode(object):
#     def __init__(self, name=None, start_idx=0, value=None, prefix=""):
#         # Node indexs
#         self._name = name if name is not None else str(start_idx)
#         self._prefix = prefix
#         self._start_idx = start_idx
#         self._end_idx = start_idx
#         self._cur_level_idx = -1
#         self._sub_nodes = OrderedDict()
#         self.attrs = dict()
#         self._is_leaf = False

#         if isinstance(value, tuple):
#             self._type = IONodeType.TUPLE
#             self._value = None
#             for idx, item in enumerate(value):
#                 subnode_prefix = (
#                     self._prefix
#                     + ("." if self._prefix else "")
#                     + str(self._cur_level_idx + 1)
#                 )
#                 self.__add_sub_node(
#                     IONode(None, self._end_idx + 1, item, subnode_prefix)
#                 )
#         elif isinstance(value, list):
#             self._type = IONodeType.LIST
#             self._value = None
#             for idx, item in enumerate(value):
#                 subnode_prefix = (
#                     self._prefix
#                     + ("." if self._prefix else "")
#                     + str(self._cur_level_idx + 1)
#                 )
#                 self.__add_sub_node(
#                     IONode(None, self._end_idx + 1, item, subnode_prefix)
#                 )
#         elif isinstance(value, dict):
#             self._type = IONodeType.DICT
#             self._value = None
#             for idx, (key, item) in enumerate(value.items()):
#                 subnode_prefix = (
#                     self._prefix
#                     + ("." if self._prefix else "")
#                     + str(self._cur_level_idx + 1)
#                 )
#                 self.__add_sub_node(
#                     IONode(key, self._end_idx + 1, item, subnode_prefix)
#                 )
#         elif isinstance(value, Tensor):
#             self._type = IONodeType.TENSOR
#             self._value = value
#             self._is_leaf = True
#         elif value is None:
#             self._type = IONodeType.NONE
#             self._value = value
#             self._is_leaf = True
#         else:
#             self._type = IONodeType.OPAQUE
#             self._value = value
#             self._is_leaf = True

#     def size(self):
#         return self._end_idx - self._start_idx + 1

#     def __add_sub_node(self, node):
#         self._sub_nodes[self._cur_level_idx + 1] = node
#         self._end_idx += node.size()
#         self._cur_level_idx += 1

#     def named_nodes(self, memo=None):
#         if memo is None:
#             memo = set()
#         if self not in memo:
#             memo.add(self)
#             yield (self._prefix + "_" + str(self._name), self)
#             for (level_idx, node) in self._sub_nodes.items():
#                 if node is None:
#                     continue
#                 for n in node.named_nodes(memo):
#                     yield n

#     def __repr__(self):
#         repr_str = ""
#         repr_str += "(name: " + self._name
#         repr_str += ", idx: " + str(self._start_idx)
#         repr_str += ", type: " + self._type
#         if self._type == IONodeType.TENSOR:
#             repr_str += ", value: " + self._value._meta_repr() + ")"
#         else:
#             repr_str += ", value: " + repr(self._value) + ")"
#         return repr_str

#     def map_leaf(self, leaf_node_fn):
#         if self._type == IONodeType.TUPLE:
#             l_value = list()
#             for (name, node) in self._sub_nodes.items():
#                 l_value.append(node.map_leaf(leaf_node_fn))
#             mapped_value = tuple(l_value)
#         elif self._type == IONodeType.LIST:
#             mapped_value = list()
#             for (name, node) in self._sub_nodes.items():
#                 mapped_value.append(node.map_leaf(leaf_node_fn))
#         elif self._type == IONodeType.DICT:
#             mapped_value = dict()
#             for (name, node) in self._sub_nodes.items():
#                 mapped_value[node._name] = node.map_leaf(leaf_node_fn)
#         else:
#             # Leaf node: TENSOR/NONE/OPAQUE
#             mapped_value = leaf_node_fn(self)
#         return mapped_value
