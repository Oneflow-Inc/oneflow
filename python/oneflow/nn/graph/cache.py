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
import os
import weakref
from collections import deque, OrderedDict
from typing import Dict, Union

from oneflow.framework.args_tree import ArgsTree
from oneflow.framework.tensor import Tensor
import oneflow as flow


class LRUCache(object):
    _cnt: int = 0

    def __init__(self, cache_size, keep_the_1st=True):
        assert cache_size >= 2
        self.cache_size = cache_size
        self.hash_map = dict()
        self.keep_the_1st = keep_the_1st
        self.queue = deque()

    def is_empty(self):
        return len(self.hash_map) == 0

    def is_full(self):
        return len(self.hash_map) >= self.cache_size

    def pop(self):
        if len(self.queue) == 0:
            return None
        pop_key = self.queue.pop()
        value = self.hash_map.pop(pop_key)
        del value
        return pop_key

    def set(self, key, value):
        new_key = None
        old_key = None
        if key in self.hash_map:
            return new_key, old_key

        if self.is_full():
            old_key = self.pop()
            assert old_key is not None, f"Cache size is {self.cache_size}, at least 2."
        assert not self.is_full()

        if not (self.keep_the_1st and self.is_empty()):
            self.queue.appendleft(key)

        value._oneflow_graph_cache_order = LRUCache._cnt
        LRUCache._cnt += 1
        self.hash_map[key] = value
        new_key = key
        return new_key, old_key

    def get(self, key):
        if key in self.hash_map:
            if key in self.queue:
                self.queue.remove(key)
                self.queue.appendleft(key)
            return self.hash_map[key]

        return None

    def items(self):
        for (key, value) in self.hash_map.items():
            yield (key, value)


class AvoidRecursiveCacheCall(object):
    def __init__(self, graph) -> None:
        self._g = graph
        self._prev_flag = self._g._run_with_cache

    def __enter__(self):
        self._g._run_with_cache = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._g._run_with_cache = self._prev_flag


class GraphCache(object):
    def __init__(self, base_graph, cache_size=10, enable_graph_shared=True):
        assert base_graph is not None and isinstance(base_graph, weakref.ProxyTypes)
        self._base_graph = base_graph

        self._cache_size = cache_size
        self._cache = None

        self._enable_shared = enable_graph_shared

    def set_cache_size(self, cache_size):
        self._cache_size = cache_size

    def enable_shared(self, enabled=True):
        self._enable_shared = enabled

    def __call__(self, *args, **kwargs):
        graph = self.get_graph(*args, **kwargs)
        with AvoidRecursiveCacheCall(graph):
            return graph(*args, **kwargs)

    def runtime_state_dict(
        self, destination=None, with_eager=False,
    ) -> Dict[str, Dict[str, Union[Dict[str, Tensor], str]]]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        for (key, graph) in self._cache.items():
            with AvoidRecursiveCacheCall(graph):
                state_dict = graph.runtime_state_dict(with_eager=with_eager)
            state_dict["cache_order"] = graph._oneflow_graph_cache_order
            state_dict["cache_key"] = key
            destination[state_dict["graph_name"]] = state_dict
        return destination

    def _init_and_get_a_graph_in_cache(self, cache_key):
        self._base_graph._print(
            0,
            0,
            self._base_graph._shallow_repr()
            + f" is creating a graph cache with key {cache_key}.",
        )
        cur_is_base = False
        if self._cache.is_empty():
            # Has no graph yet
            cur_is_base = True
            graph = self._base_graph
        else:
            # Create new graph from base
            graph = self._base_graph.__class__(
                *self._base_graph._cached_init_args,
                **self._base_graph._cached_init_kwargs,
            )
            graph._run_with_cache = False
            graph._dynamic_input_graph_cache = None
            graph._cached_init_args = None
            graph._cached_init_kwargs = None

        if self._enable_shared is True:
            if cur_is_base:
                graph.enable_shared()
            else:
                graph.share_from(self._base_graph)
        new_key, old_key = self._cache.set(cache_key, graph)
        if old_key is not None:
            self._base_graph._print(
                0,
                0,
                self._base_graph._shallow_repr()
                + f" cache is full(cache size {self._cache_size}), has deleted an old graph cache with key {old_key}.",
            )
        assert new_key is not None

        return graph

    def load_runtime_state_dict(
        self, state_dict: Dict[str, Dict[str, Union[Dict[str, Tensor], str]]]
    ) -> None:
        graph_dict = dict()
        for _, sub_state_dict in state_dict.items():
            cache_order = sub_state_dict["cache_order"]
            graph_dict[cache_order] = sub_state_dict

        if self._cache is None:
            self._cache = LRUCache(self._cache_size)
        for _, sub_state_dict in sorted(graph_dict.items()):
            cache_key = sub_state_dict["cache_key"]
            graph = self._cache.get(cache_key)
            assert graph is None
            graph = self._init_and_get_a_graph_in_cache(cache_key)
            with AvoidRecursiveCacheCall(graph):
                graph.load_runtime_state_dict(sub_state_dict)

    def gen_key(self, *args, **kwargs):
        flattened_shapes = []
        args_tree = ArgsTree((args, kwargs), False)
        for arg in args_tree.iter_nodes():
            if isinstance(arg, Tensor):
                flattened_shapes.append(arg.shape)
        return tuple(flattened_shapes)

    def get_graph(self, *args, **kwargs):
        if self._cache is None:
            self._cache = LRUCache(self._cache_size)

        cache_key = hash(self.gen_key(*args, **kwargs))
        graph = self._cache.get(cache_key)

        # Create graph
        if graph is None:
            self._base_graph._print(
                0,
                0,
                self._base_graph._shallow_repr()
                + " got a new input shape, is compiling a new graph.",
            )
            graph = self._init_and_get_a_graph_in_cache(cache_key)

        return graph
