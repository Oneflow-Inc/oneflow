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


def seq_to_func_return(li):
    if len(li) == 0:
        return None
    elif len(li) == 1:
        return li[0]
    else:
        return tuple(li)

class IONodeType:
    EMPTY = "EMPTY"
    TENSOR = "TENSOR"
    NONE = "NONE"
    LIST = "LIST"
    TUPLE = "TUPLE"
    DICT = "DICT"
    # Opaque data type
    OPA= "OPA"

class IONode(object):
    def __init__(self, name=None, start_idx=0, n_type=IONodeType.EMPTY, value=None, seq=None, dic=None):
        # Node indexs
        self._name = name if name is not None else str(start_idx)
        self._start_idx = start_idx
        self._end_idx = start_idx
        self._cur_level_idx = 0

        # This node
        self._type = n_type
        self._value = value

        # Sub nodes
        self._sub_nodes = OrderedDict()
        self._seq = seq
        self._dic = dic

        if self._seq is not None:
            for idx, item in enumerate(self._seq):
                self.__add_sub_item(None, item)

        if self._dic is not None:
            for idx, (key, item) in enumerate(self._dic.items()):
                self.__add_sub_item(key, item)

    def size(self):
        return self._end_idx - self._start_idx + 1
    
    def __add_sub_node(self, node):
        self._sub_nodes[self._cur_level_idx + 1] = node
        self._end_idx += node.size()
        self._cur_level_idx += 1

    def __add_sub_item(self, key, item):
        if isinstance(item, tuple):
            self.__add_sub_node(IONode(key, self._end_idx + 1, IONodeType.TUPLE, None, item, None))
        elif isinstance(item, list):
            self.__add_sub_node(IONode(key, self._end_idx + 1, IONodeType.LIST, None, item, None))
        elif isinstance(item, dict):
            self.__add_sub_node(IONode(key, self._end_idx + 1, IONodeType.DICT, None, None, item))
        elif isinstance(item, Tensor):
            self.__add_sub_node(IONode(key, self._end_idx + 1, IONodeType.TENSOR, item, None, None))
        elif item is None:
            self.__add_sub_node(IONode(key, self._end_idx + 1, IONodeType.NONE, item, None, None))
        else:
            self.__add_sub_node(IONode(key, self._end_idx + 1, IONodeType.OPA, item, None, None))

    def named_nodes(self, memo = None, prefix: str = ""):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield (prefix + "-" + str(self._name), self)
            for (name, node) in self._sub_nodes.items():
                if node is None:
                    continue
                subnode_prefix = prefix + ("." if prefix else "") + str(name)
                for n in node.named_nodes(memo, subnode_prefix):
                    yield n
    
    def __repr__(self):
        repr_str = ""
        repr_str += "(name: " + self._name
        repr_str += ', idx: ' + str(self._start_idx)
        repr_str += ", type: " + self._type
        repr_str += ", value: " + repr(self._value) + ")"
        return repr_str


    def mapping_tensor(self, fn):
        pass

    def to_py_arg():
        pass


