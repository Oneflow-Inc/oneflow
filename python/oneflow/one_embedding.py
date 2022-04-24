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
import oneflow as flow
from oneflow.nn.module import Module
import json
import datetime
from oneflow._oneflow_internal import OneEmbeddingHandler
import numpy as np
import traceback


def _check_initializer(initializer):
    assert isinstance(initializer, dict)
    assert initializer.__contains__("type")
    initializer_type = initializer["type"]
    assert initializer_type in ["uniform", "normal"]
    if initializer_type == "uniform":
        assert initializer.__contains__("low")
        assert initializer.__contains__("high")
    elif initializer_type == "normal":
        assert initializer.__contains__("mean")
        assert initializer.__contains__("std")
    else:
        raise NotImplementedError("unsupported initializer_type")


def _check_cache(cache):
    assert isinstance(cache, dict)
    assert cache.__contains__("policy")
    assert cache["policy"] in ["lru", "full"]
    cache_memory_budget_mb = 0
    if cache.__contains__("cache_memory_budget_mb"):
        cache_memory_budget_mb = cache["cache_memory_budget_mb"]
    capacity = 0
    if cache.__contains__("capacity"):
        capacity = cache["capacity"]
    assert cache_memory_budget_mb > 0 or capacity > 0
    assert cache.__contains__("value_memory_kind")
    assert cache["value_memory_kind"] in ["device", "host"]


class MultiTableEmbedding(Module):
    r"""MultiTableEmbedding represent multi Embedding tables with same embedding_dim, dtype, and key_type.

    Args:
        name (str): The name of Embedding
        embedding_dim (int): the size of each embedding vector
        dtype (flow.dtype): the data type of embeddings
        key_type (flow.dtype): the data type of feature ids
        tables (list): list of table param which can be made by flow.one_embedding.make_table_options
        store_options (dict): store option of Embedding
        default_initializer (dict, optional): if tables param is None, use default_initializer to initialize table. Defaults to None.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> import oneflow.nn as nn
        >>> # a simple example with 3 table
        >>> table_size_array = [39884407, 39043, 17289]
        >>> vocab_size = sum(table_size_array)
        >>> num_tables = len(table_size_array)
        >>> embedding_size = 128
        >>> scales = np.sqrt(1 / np.array(table_size_array))
        >>> tables = [
        >>>     flow.one_embedding.make_table_options(
        >>>         flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        >>>     )
        >>>     for scale in scales
        >>> ]
        >>> store_options = flow.one_embedding.make_cached_ssd_store_options(
        >>>     cache_budget_mb=8192, persistent_path="/your_path_to_ssd", capacity=vocab_size,
        >>> )
        >>> embedding = flow.one_embedding.MultiTableEmbedding(
        >>>     name="my_embedding",
        >>>     embedding_dim=embedding_size,
        >>>     dtype=flow.float,
        >>>     key_type=flow.int64,
        >>>     tables=tables,
        >>>     store_options=store_options,
        >>> )
        >>> embedding.to("cuda")
        >>> mlp = flow.nn.FusedMLP(
        >>>     in_features=embedding_size * num_tables,
        >>>     hidden_features=[512, 256, 128],
        >>>     out_features=1,
        >>>     skip_final_activation=True,
        >>> )
        >>> mlp.to("cuda")
        >>>
        >>> class TrainGraph(flow.nn.Graph):
        >>>     def __init__(self,):
        >>>         super().__init__()
        >>>         self.embedding_lookup = embedding
        >>>         self.mlp = mlp
        >>>         self.add_optimizer(
        >>>             flow.optim.SGD(self.embedding_lookup.parameters(), lr=0.1, momentum=0.0)
        >>>         )
        >>>         self.add_optimizer(
        >>>             flow.optim.SGD(self.mlp.parameters(), lr=0.1, momentum=0.0)
        >>>         ) 
        >>>     def build(self, ids):
        >>>         embedding = self.embedding_lookup(ids)
        >>>         loss = self.mlp(flow.reshape(embedding, (-1, num_tables * embedding_size)))
        >>>         loss = loss.sum()
        >>>         loss.backward()
        >>>         return loss 
        >>> ids = np.random.randint(0, 1000, (100, num_tables), dtype=np.int64)
        >>> ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
        >>> graph = TrainGraph()
        >>> loss = graph(ids_tensor)
        >>> print(loss)

    """

    def __init__(
        self,
        name,
        embedding_dim,
        dtype,
        key_type,
        tables,
        store_options,
        default_initializer=None,
    ):
        super().__init__()
        default_initializer = default_initializer or {
            "type": "normal",
            "mean": 0,
            "std": 0.05,
        }
        key_value_store_options = {}
        embedding_tables = {}
        key_value_store_options["name"] = name
        assert embedding_dim > 0
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.key_type = key_type
        parallel_num = flow.env.get_world_size()

        key_type_size = np.dtype(
            flow.convert_oneflow_dtype_to_numpy_dtype(key_type)
        ).itemsize
        assert key_type_size > 0
        key_value_store_options["key_type_size"] = key_type_size
        value_type_size = np.dtype(
            flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
        ).itemsize
        assert value_type_size > 0
        key_value_store_options["value_type_size"] = value_type_size

        scale_factor = store_options["size_factor"]
        key_value_store_options["storage_dim"] = scale_factor * embedding_dim

        # kv store
        assert store_options.__contains__("kv_store")
        kv_store = store_options["kv_store"]
        assert isinstance(kv_store, dict)
        if kv_store.__contains__("caches"):
            caches = kv_store["caches"]
            assert isinstance(caches, (dict, list, tuple))
            if isinstance(caches, dict):
                _check_cache(caches)
                caches = [caches]
            else:
                assert len(caches) <= 2
                for i in range(len(caches)):
                    assert isinstance(caches[i], dict)
                    _check_cache(caches[i])
            for i in range(len(caches)):
                if caches[i].__contains__("capacity"):
                    caches[i]["capacity"] = caches[i]["capacity"] // parallel_num
        assert kv_store.__contains__("persistent_table")
        persistent_table = kv_store["persistent_table"]
        assert isinstance(persistent_table, dict)
        assert persistent_table.__contains__("path")
        persistent_table_path = persistent_table["path"]
        assert isinstance(persistent_table_path, (str, list, tuple))
        if isinstance(persistent_table_path, (list, tuple)):
            assert len(persistent_table_path) == parallel_num
        if persistent_table.__contains__("physical_block_size"):
            assert persistent_table["physical_block_size"] in [512, 4096]
        else:
            persistent_table["physical_block_size"] = 4096
        if persistent_table.__contains__("capacity_hint"):
            assert persistent_table["capacity_hint"] >= 0
            persistent_table["capacity_hint"] = (
                persistent_table["capacity_hint"] // parallel_num
            )

        key_value_store_options["kv_store"] = kv_store

        # initializer
        if tables is not None:
            assert isinstance(tables, (list, tuple))
            for table in tables:
                assert isinstance(table, dict)
                assert table.__contains__("initializer")
                _check_initializer(table["initializer"])
            # TODO(guoran): change "columns" to "tables" and modify c++ code
            embedding_tables["columns"] = tables
        else:
            assert default_initializer is not None
            _check_initializer(default_initializer)
            embedding_tables["columns"] = [{"initializer": default_initializer}]
        key_value_store_options["parallel_num"] = parallel_num
        self.key_value_store_options = json.dumps(key_value_store_options)
        self.embedding_tables = json.dumps(embedding_tables)
        self.num_tables = len(embedding_tables["columns"])
        self.local_rank = flow.env.get_local_rank()
        self.rank_id = flow.env.get_rank()
        self.world_size = parallel_num
        self.handler = OneEmbeddingHandler(
            self.key_value_store_options, self.local_rank, self.rank_id, self.world_size
        )
        self.shadow = flow.nn.Parameter(flow.Tensor(1))

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        snapshot_timestamp_tensor = flow.tensor(
            datetime.datetime.now().timestamp(), dtype=flow.float64, device="cuda"
        )
        # Broadcast timestamp tensor from master rank.
        flow.comm.broadcast(snapshot_timestamp_tensor, src=0)
        snapshot_timestamp = float(snapshot_timestamp_tensor.numpy())
        snapshot_timestamp_datetime = datetime.datetime.fromtimestamp(
            snapshot_timestamp
        )
        snapshot_timestamp_str = snapshot_timestamp_datetime.strftime(
            "%Y-%m-%d-%H-%M-%S-%f"
        )
        self.handler.SaveSnapshot(snapshot_timestamp_str)
        destination[prefix + "OneEmbedding"] = snapshot_timestamp_str

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "OneEmbedding"
        if key in state_dict:
            saved_snapshot_name = state_dict[key]
            try:
                self.handler.LoadSnapshot(saved_snapshot_name)
            except Exception as ex:
                error_msgs.append(
                    'While Loading OneEmbedding Snapshot named "{}" failed, please check whether the Snapshot exist'.format(
                        saved_snapshot_name
                    )
                )

    def save_snapshot(self, snapshot_name):
        """save snapshot

        Args:
            snapshot_name (str): the snapshot_name, snapshot will be saved in the snapshots dir under your_configed_persistent_path
    
        For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> # use embedding create by flow.one_embedding.MultiTableEmbedding
            >>> embedding.save_snapshot("my_snapshot1")
            >>> # a snapshot named "my_snapshot1" have been saved in the "snapshots" dir under your_configed_persistent_path
            >>> # which can be reload by flow.one_embedding.load_snapshot
        """
        self.handler.SaveSnapshot(snapshot_name)

    def load_snapshot(self, snapshot_name):
        """load snapshot

        Args:
            snapshot_name (str): the snapshot_name, snapshot will be load from your_configed_persistent_path
    
        For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> # use embedding create by flow.one_embedding.MultiTableEmbedding
            >>> embedding.load_snapshot("my_snapshot1")
            >>> # load a snapshot named "my_snapshot1" from your_configed_persistent_path
        """
        self.handler.LoadSnapshot(snapshot_name)

    def forward(self, ids, table_ids=None):
        """forward of MultiTableEmbedding

        Args:
            ids (flow.tensor): the feature ids
            table_ids (flow.tensor, optional): the table_id of each id, must be same shape as ids. There is no need to pass table_ids, if has config only one table or the ids has shape (batch_size, num_tables), and each column's id belongs to the column_id th table, otherwise, you should pass the tensor_ids.

        Returns:
            flow.tensor: the result of embedding lookup
        """
        assert self.key_type == ids.dtype, "ids data_type must equals key_type"
        return flow._C.one_embedding_lookup(
            self.shadow,
            ids,
            table_ids,
            self.dtype,
            self.embedding_dim,
            self.num_tables,
            self.embedding_tables,
            self.key_value_store_options,
        )


def make_device_mem_store_options(
    persistent_path, capacity, size_factor=1, physical_block_size=512
):
    """make GPU only store_options param of MultiTableEmbedding

    Args:
        persistent_path (str, list): persistent storage path of Embedding. If passed a str, current rank Embedding will be saved in path/rank_id-num_ranks path. If passed a list, the list length must equals num_ranks, each elem of list represent the path of rank_id Embedding.
        capacity (int): total capacity of Embedding
        size_factor (int, optional): store size factor of embedding_dim, if SGD update, and momentum = 0, should be 1, if momentum > 0, it should be 2. if Adam, should be 3. Defaults to 1.
        physical_block_size (int, optional): physical_block_size should be sector size. Defaults to 512.

    Returns:
        dict: GPU only store_options param of MultiTableEmbedding

    See also :func:`oneflow.one_embedding.make_cached_ssd_store_options`
    """

    assert isinstance(persistent_path, (str, list, tuple))
    assert capacity > 0
    options = {
        "kv_store": {
            "caches": [
                {
                    "policy": "full",
                    "capacity": int(capacity),
                    "value_memory_kind": "device",
                }
            ],
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
                "capacity_hint": int(capacity),
            },
        },
        "size_factor": size_factor,
    }
    return options


def make_cached_ssd_store_options(
    cache_budget_mb,
    persistent_path,
    capacity=None,
    size_factor=1,
    physical_block_size=512,
):
    """make SSD use GPU as cache store_options param of MultiTableEmbedding

    Args:
        cache_budget_mb (int): the MB budget of per GPU as cache.
        persistent_path (str, list): persistent storage path of Embedding, must use fast SSD because of frequently random disk access during training. If passed a str, current rank Embedding will be saved in path/rank_id-num_ranks path. If passed a list, the list length must equals num_ranks, each elem of list represent the path of rank_id Embedding.
        capacity (int): total capacity of Embedding
        size_factor (int, optional): store size factor of embedding_dim, if SGD update, and momentum = 0, should be 1, if momentum > 0, it should be 2. if Adam, should be 3. Defaults to 1.
        physical_block_size (int, optional): physical_block_size should be sector size. Defaults to 512.

    Returns:
        dict: SSD use GPU as cache store_options param of MultiTableEmbedding

    For example:

    .. code-block:: python

        >>> import oneflow as flow    
        >>> store_options = flow.one_embedding.make_cached_ssd_store_options(
        >>>     cache_budget_mb=8192, persistent_path="/your_path_to_ssd", capacity=vocab_size,
        >>> )
        >>> # pass the store_options to the "store_options" param of flow.one_embedding.MultiTableEmbedding
        >>> # ...
    """
    assert isinstance(persistent_path, (str, list, tuple))
    assert cache_budget_mb > 0
    if capacity is not None:
        assert capacity > 0
    else:
        capacity = 0
    options = {
        "kv_store": {
            "caches": [
                {
                    "policy": "lru",
                    "cache_memory_budget_mb": cache_budget_mb,
                    "value_memory_kind": "device",
                }
            ],
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
                "capacity_hint": int(capacity),
            },
        },
        "size_factor": size_factor,
    }
    return options


def make_cached_host_mem_store_options(
    cache_budget_mb, persistent_path, capacity, size_factor=1, physical_block_size=512,
):
    """make host use GPU as cache store_options param of MultiTableEmbedding

    Args:
        cache_budget_mb (int): the MB budget of per GPU as cache.
        persistent_path (str, list): persistent storage path of Embedding. If passed a str, current rank Embedding will be saved in path/rank_id-num_ranks path. If passed a list, the list length must equals num_ranks, each elem of list represent the path of rank_id Embedding.
        capacity (int): total capacity of Embedding
        size_factor (int, optional): store size factor of embedding_dim, if SGD update, and momentum = 0, should be 1, if momentum > 0, it should be 2. if Adam, should be 3. Defaults to 1.
        physical_block_size (int, optional): physical_block_size should be sector size. Defaults to 512.

    Returns:
        dict: host use GPU as cache store_options param of MultiTableEmbedding

    See also :func:`oneflow.one_embedding.make_cached_ssd_store_options`
    """
    assert isinstance(persistent_path, (str, list, tuple))
    assert cache_budget_mb > 0
    assert capacity > 0
    options = {
        "kv_store": {
            "caches": [
                {
                    "policy": "lru",
                    "cache_memory_budget_mb": cache_budget_mb,
                    "value_memory_kind": "device",
                },
                {
                    "policy": "full",
                    "capacity": int(capacity),
                    "value_memory_kind": "host",
                },
            ],
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
                "capacity_hint": int(capacity),
            },
        },
        "size_factor": size_factor,
    }
    return options


def make_uniform_initializer(low, high):
    """make uniform initializer param of make_table_options

    Args:
        low (float): A python scalar. Lower bound of the range of random values to generate.
        high (float): A python scalar. Upper bound of the range of random values to generate.

    Returns:
        dict: initializer param of make_table_options
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        >>> # pass the initializer to flow.one_embedding.make_table_options
        >>> # ...
    """
    return {"type": "uniform", "low": low, "high": high}


def make_normal_initializer(mean, std):
    """make normal initializer param of make_table_options

    Args:
        mean (float): A python scalar. Mean of the random values to generate.
        std (float): A python scalar. Standard deviation of the random values to generate.

    Returns:
        dict: initializer param of make_table_options
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_normal_initializer(mean=0, std=0.01)
        >>> # pass the initializer to flow.one_embedding.make_table_options
        >>> # ...
    """
    return {"type": "normal", "mean": mean, "std": std}


def make_table_options(initializer):
    """make table param of MultiTableEmbedding tables

    Args:
        initializer (dict): initializer param, make by make_uniform_initializer or make_normal_initializer

    Returns:
        dict: table param of MultiTableEmbedding tables
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        >>> table1 = flow.one_embedding.make_table_options(initializer)
        >>> table2 = flow.one_embedding.make_table_options(initializer)
        >>> tables = [table1, table2]
        >>> # pass the tables to the "tables" param of flow.one_embedding.MultiTableEmbedding
        >>> # ...
        
    """
    return {"initializer": initializer}


def make_table(initializer):
    """alias of `oneflow.one_embedding.make_table_options`

    See also :func:`oneflow.one_embedding.make_table_options`
    """
    return make_table_options(initializer)
