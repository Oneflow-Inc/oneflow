import os
import weakref
from collections import deque

from oneflow.framework.args_tree import ArgsTree
from oneflow.framework.tensor import Tensor
import oneflow as flow

class OneFlowGraph(object):
    def __init__(self, graph_class, *args, **kwargs):
        self.graph_ = graph_class(*args, **kwargs)
        self.is_compiled_ = False
        self.is_shared_from_ = False

    @property
    def is_compiled(self):
        return self.is_compiled_

    def compile(self, *args, **kwargs):
        if self.is_compiled_:
            return

        global_class_name = self.graph_.__class__.__name__
        compilation_time = 0
        if self.is_shared_from_:
            self.graph_._compile_from_shared(*args, **kwargs)
        else:
            self.graph_._compile(*args, **kwargs)

        self.is_compiled_ = True

    def load_runtime_state_dict(self, state_dict):
        if self.is_compiled_:
            return

        global_class_name = self.graph_.__class__.__name__
        self.graph_.load_runtime_state_dict(state_dict)
        self.is_compiled_ = True

    def share_from(self, other_graph):
        self.graph_.share_from(other_graph.graph_)
        self.is_shared_from_ = True

    def __call__(self, *args, **kwargs):
        if not self.is_compiled_:
            self.compile(*args, **kwargs)

        return self.graph_(*args, **kwargs)


class LRUCache(object):
    _cnt: int = 0
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.queue = deque()
        self.hash_map = dict()

    def front(self):
        if self.is_empty():
            return None

        key = self.queue[0]
        return self.hash_map[key]

    def is_empty(self):
        return len(self.queue) == 0

    def is_queue_full(self):
        return len(self.queue) >= self.cache_size

    def pop(self):
        pop_key = self.queue.pop()
        value = self.hash_map.pop(pop_key)
        del value
        return pop_key

    def set(self, key, value):
        if key in self.hash_map:
            return None

        pop_key = None
        while self.is_queue_full():
            pop_key = self.pop()

        self.queue.appendleft(key)
        value._oneflow_graph_cache_order = LRUCache._cnt
        LRUCache._cnt += 1
        self.hash_map[key] = value
        return pop_key if pop_key is not None else key

    def get(self, key):
        if key in self.hash_map:
            self.queue.remove(key)
            self.queue.appendleft(key)
            return self.hash_map[key]

        return None
    
    def pairs(self):
        for (key, value) in self.hash_map.items():
            yield (key, value)



class GraphCache(object):
    def __init__(self, base_graph, cache_size=10, enable_graph_shared=True):
        assert base_graph is not None and isinstance(base_graph, weakref.ProxyTypes)
        self._base_graph = base_graph

        self._cache_size = cache_size
        self._cache = None

        self._enable_shared = enable_graph_shared

        self._enable_save_graph = False
        self._graph_save_load_path = None

    def set_cache_size(self, cache_size):
        self._cache_size = cache_size

    def enable_shared(self, enabled=True):
        self._enable_shared= enabled
    
    def enable_save_graph(self, enabled=True):
        self._enable_save_graph = enabled

    def __call__(self, *args, **kwargs):
        graph = self.get_graph(*args, **kwargs)
        if graph._run_with_cache == True:
            graph._run_with_cache = False
            output = graph(*args, **kwargs)
            graph._run_with_cache = True
            return output
        else:
            return graph(*args, **kwargs)

    def save_graph(self, path):
        if self.enable_save_graph_:
            for (graph_class_name, cache) in self.cache_bucket_.items():
                for (key, graph) in cache.pairs():
                    state_dict = graph.graph_.runtime_state_dict()
                    state_dict["cache_order"] = graph._oneflow_graph_cache_order
                    state_dict["cache_key"] = key
                    state_dict["graph_class_name"] = graph_class_name
                    flow.save(state_dict, os.path.join(path, graph_class_name + "_" + str(hash(key))))

    def load_graph(self, path, graph_class2init_args=None):
        sub_files = [ f.path for f in os.scandir(path) if f.is_file() ]
        graph_dict = dict()
        for sub_file in sub_files:
            state_dict = flow.load(sub_file)
            cache_order = state_dict["cache_order"]
            graph_dict[cache_order] = state_dict
        
        for order, state_dict in sorted(graph_dict.items()):
            graph_class_name  = state_dict["graph_class_name"]
            cache_key = state_dict["cache_key"]
            if graph_class_name not in self.cache_bucket_:
                self.cache_bucket_[graph_class_name] = LRUCache(self.cache_size_)
            compile_cache = self.cache_bucket_[graph_class_name]
            if graph_class_name in graph_class2init_args:
                init_args = graph_class2init_args[graph_class_name]
                graph = OneFlowGraph(init_args[0], init_args[1])
            else:
                graph = OneFlowGraph(flow.nn.Graph)
            if self.enable_share_mem_ is True:
                if graph_class_name in self.share_origin_:
                    graph.share_from(self.share_origin_[graph_class_name])
                else:
                    self.share_origin_[graph_class_name] = graph
                    graph.graph_.enable_shared()

            graph.load_runtime_state_dict(state_dict)
            ret = compile_cache.set(cache_key, graph)
            assert ret is not None
    
    def gen_key(self, *args, **kwargs):
        flattened_shapes = []
        args_tree = ArgsTree((args, kwargs), False)
        for arg in args_tree.iter_nodes():
            if isinstance(arg, Tensor):
                flattened_shapes.append(arg.shape)
        return tuple(flattened_shapes)


    def get_graph(self, *args, **kwargs):
        if self._cache is None:
            self._cache =  LRUCache(self._cache_size)

        cache_key = hash(self.gen_key(*args, **kwargs))
        graph = self._cache.get(cache_key)

        # Create graph
        if graph is None:
            cur_is_base = False
            if self._cache.is_empty():
                # Has no graph yet
                cur_is_base = True
                graph = self._base_graph
                print("get base", cache_key)
            else:
                # Create new graph from base
                graph = self._base_graph.__class__(*self._base_graph._cached_init_args, **self._base_graph._cached_init_kwargs)
                graph._run_with_cache = False
                graph._dynamic_input_graph_cache = None
                graph._cached_init_args = None
                graph._cached_init_kwargs = None
                print("get new", cache_key)
            ret = self._cache.set(cache_key, graph)
            assert ret is not None

            if self._enable_save_graph:
                graph.enable_save_runtime_state_dict()
            if self._enable_shared is True:
                if cur_is_base:
                    graph.enable_shared()
                else:
                    graph.share_from(self._base_graph)
        else:
            print("====> hit cache ", cache_key)

        return graph