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
from typing import Callable, Dict, Iterator, List, Union
import oneflow as flow
from oneflow.nn.modules.module import Module
from oneflow.optim.optimizer import Optimizer
from oneflow.nn.parameter import Parameter
import json
import datetime
from oneflow._oneflow_internal import OneEmbeddingHandler
from oneflow._oneflow_internal import PersistentTableReader
from oneflow._oneflow_internal import PersistentTableWriter
import numpy as np
import traceback
from oneflow import nn
import oneflow.framework.graph_build_util as graph_build_util


def _check_initializer(initializer):
    assert isinstance(initializer, dict)
    assert initializer.__contains__("type")
    initializer_type = initializer["type"]
    assert initializer_type in ["uniform", "normal", "constant", "trunc_normal"]
    if initializer_type == "uniform":
        assert initializer.__contains__("low")
        assert initializer.__contains__("high")
    elif initializer_type == "normal":
        assert initializer.__contains__("mean")
        assert initializer.__contains__("std")
    elif initializer_type == "constant":
        assert initializer.__contains__("value")
    elif initializer_type == "trunc_normal":
        assert initializer.__contains__("mean")
        assert initializer.__contains__("std")
        assert initializer.__contains__("a")
        assert initializer.__contains__("b")
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


def _init(
    name, embedding_dims, dtype, key_type, tables, store_options, default_initializer
):
    default_initializer = default_initializer or {
        "type": "normal",
        "mean": 0,
        "std": 0.05,
    }
    key_value_store_options = {}
    embedding_tables = {}
    key_value_store_options["name"] = name

    if isinstance(embedding_dims, (list, tuple)):
        column_dims = embedding_dims
        embedding_dim = sum(embedding_dims)
    else:
        assert embedding_dims > 0
        column_dims = [embedding_dims]
        embedding_dim = embedding_dims
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
    key_value_store_options["value_type"] = str(dtype)
    scale_factor = store_options["size_factor"]
    storage_dim = store_options["storage_dim"]
    if storage_dim != -1:
        key_value_store_options["storage_dim"] = storage_dim
    else:
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
        for i in range(len(tables)):
            table = tables[i]
            if table.__contains__("columns"):
                assert not table.__contains__("initializer")
                columns = table["columns"]
                assert len(columns) == len(column_dims)
                for column in columns:
                    assert isinstance(column, dict)
                    assert column.__contains__("initializer")
                    _check_initializer(column["initializer"])
            else:
                assert isinstance(table, dict)
                assert table.__contains__("initializer")
                _check_initializer(table["initializer"])
                columns = []
                for j in range(len(column_dims)):
                    columns.append(make_column_options(table["initializer"]))
                table["columns"] = columns
                del table["initializer"]
        embedding_tables["tables"] = tables
    else:
        assert default_initializer is not None
        _check_initializer(default_initializer)
        columns = []
        for j in range(len(column_dims)):
            columns.append(make_column_options(default_initializer))
        embedding_tables["tables"] = [{"columns": columns}]
    embedding_tables["column_dims"] = column_dims
    key_value_store_options["parallel_num"] = parallel_num
    return embedding_dim, embedding_tables, key_value_store_options


class Embedding(Module):
    def __init__(
        self,
        name,
        embedding_dim,
        dtype,
        key_type,
        tables,
        store_options,
        default_initializer=None,
        padding_idx=None,
        seed=0,
    ):
        super().__init__()
        self.dtype = dtype
        self.key_type = key_type
        parallel_num = flow.env.get_world_size()
        self.embedding_dim, embedding_tables, key_value_store_options = _init(
            name,
            embedding_dim,
            dtype,
            key_type,
            tables,
            store_options,
            default_initializer,
        )
        self.storage_dim = key_value_store_options["storage_dim"]
        self.embedding_name = key_value_store_options["name"]
        self.seed = seed
        self.is_full_cache = (
            len(key_value_store_options["kv_store"]["caches"]) > 0
            and key_value_store_options["kv_store"]["caches"][0]["policy"] == "full"
        )
        self.key_value_store_options = json.dumps(key_value_store_options)
        self.embedding_tables = json.dumps(embedding_tables)
        self.num_tables = len(embedding_tables["tables"])
        self.local_rank = flow.env.get_local_rank()
        self.rank_id = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        self.handler = OneEmbeddingHandler(
            self.key_value_store_options, self.local_rank, self.rank_id, self.world_size
        )

        self.shadow = flow.nn.Parameter(flow.Tensor(1))
        self.padding_idx = padding_idx
        self.embedding = None

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
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
        destination[prefix + "OneEmbeddingSnapshot"] = snapshot_timestamp_str
        destination[
            prefix + "OneEmbeddingKeyValueOptions"
        ] = self.key_value_store_options

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
        key = prefix + "OneEmbeddingSnapshot"
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
        """Embedding lookup operation

        Args:
            ids (flow.tensor): the feature ids
            table_ids (flow.tensor, optional): the table_id of each id, must be same shape as ids. There is no need to pass table_ids, if has config only one table or the ids has shape (batch_size, num_tables), and each column's id belongs to the column_id th table, otherwise, you should pass the tensor_ids.

        Returns:
            flow.tensor: the result of embedding lookup
        """
        assert self.key_type == ids.dtype, "ids data_type must equals key_type"
        embedding = flow._C.one_embedding_fused_lookup(
            self.shadow,
            ids,
            table_ids,
            self.dtype,
            self.embedding_name,
            self.storage_dim,
            self.embedding_dim,
            self.is_full_cache,
            self.num_tables,
            self.embedding_tables,
            self.padding_idx,
            self.seed,
        )
        if embedding.requires_grad and not graph_build_util.lazy_mode.is_enabled():
            if self.embedding is not None:
                raise ValueError(
                    "You are training without set embedding optimizer, Please add flow.one_embedding.Optimizer after optimizer."
                )

            self.embedding = embedding
            self.embedding.retain_grad()
            self.ids = ids
            self.table_ids = table_ids
        return embedding

    def shuffle_and_lookup(self, state_initializer):
        embedding_grad = self.embedding.grad
        if self.world_size > 1:
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                cur_rank_num_unique,
                cur_rank_unique_ids,
                cur_rank_unique_table_ids,
                cur_rank_inverse_indices,
            ) = flow._C.one_embedding_id_shuffle(
                self.ids, self.table_ids, self.num_tables, self.embedding_name
            )
            unique_values = flow._C.one_embedding_lookup(
                cur_rank_num_unique,
                cur_rank_unique_ids,
                cur_rank_unique_table_ids,
                self.dtype,
                self.dtype,
                self.storage_dim,
                self.embedding_dim,
                self.embedding_name,
                self.embedding_tables,
                state_initializer,
                seed=self.seed,
            )
            cur_rank_unique_embedding_grad = flow._C.one_embedding_embedding_gradient_shuffle(
                embedding_grad,
                num_unique_matrix,
                cur_rank_inverse_indices,
                inverse_unique_partition_indices,
                self.embedding_name,
            )
        else:
            (
                cur_rank_num_unique,
                cur_rank_unique_ids,
                cur_rank_unique_table_ids,
                inverse_indices,
            ) = flow._C.one_embedding_unique_key_value_pair(
                self.ids, self.table_ids, self.num_tables, self.embedding_name
            )
            unique_values = flow._C.one_embedding_lookup(
                cur_rank_num_unique,
                cur_rank_unique_ids,
                cur_rank_unique_table_ids,
                self.dtype,
                self.dtype,
                self.storage_dim,
                self.embedding_dim,
                self.embedding_name,
                self.embedding_tables,
                state_initializer,
                seed=self.seed,
            )
            cur_rank_unique_embedding_grad = flow._C.unsorted_segment_sum(
                embedding_grad,
                inverse_indices,
                axis=0,
                num_segments=unique_values.shape[0],
            )
        self.embedding = None
        return (
            cur_rank_num_unique,
            cur_rank_unique_ids,
            unique_values,
            cur_rank_unique_embedding_grad,
        )

    def sgd_update(self, param_group, step):
        lr = param_group["lr"]
        l2 = param_group["weight_decay"]
        momentum = param_group["momentum"]
        (
            cur_rank_num_unique,
            cur_rank_unique_ids,
            unique_values,
            cur_rank_unique_embedding_grad,
        ) = self.shuffle_and_lookup("")
        updated_values = flow._C.one_embedding_sgd_update(
            cur_rank_num_unique,
            unique_values,
            cur_rank_unique_embedding_grad,
            learning_rate_val=lr,
            scale=1.0,
            weight_decay=l2,
            momentum=momentum,
            line_size=self.storage_dim,
            embedding_size=self.embedding_dim,
            embedding_name=self.embedding_name,
        )
        flow._C.one_embedding_embedding_put(
            cur_rank_num_unique,
            cur_rank_unique_ids,
            updated_values,
            self.embedding_name,
            self.storage_dim,
        )

    def adam_update(self, param_group, step):
        line_size = self.storage_dim
        embedding_size = self.embedding_dim
        lr = param_group["lr"]
        # not adjust, because it has been set in optimizer's step
        bias_correction1 = param_group["bias_correction1"]
        bias_correction2 = param_group["bias_correction2"]
        l2 = param_group["weight_decay"]
        beta1 = param_group["betas"][0]
        beta2 = param_group["betas"][1]
        epsilon = param_group["eps"]
        do_bias_correction = param_group["do_bias_correction"]
        amsgrad = param_group["amsgrad"]
        assert amsgrad == False, "one_embedding's adam not support amsgrad"
        state_initializer = [make_constant_initializer(0), make_constant_initializer(0)]
        (
            cur_rank_num_unique,
            cur_rank_unique_ids,
            unique_values,
            cur_rank_unique_embedding_grad,
        ) = self.shuffle_and_lookup(json.dumps(state_initializer))
        updated_values = flow._C.one_embedding_adam_update(
            cur_rank_num_unique,
            unique_values,
            cur_rank_unique_embedding_grad,
            learning_rate_val=lr,
            scale=1.0,
            weight_decay=l2,
            beta1=beta1,
            beta2=beta2,
            bias_correction1_val=bias_correction1,
            bias_correction2_val=bias_correction2,
            epsilon=epsilon,
            do_bias_correction=do_bias_correction,
            line_size=line_size,
            embedding_size=embedding_size,
            embedding_name=self.embedding_name,
        )
        flow._C.one_embedding_embedding_put(
            cur_rank_num_unique,
            cur_rank_unique_ids,
            updated_values,
            self.embedding_name,
            line_size,
        )

    def adagrad_update(self, param_group, step):
        lr = param_group["lr"]
        l2 = param_group["weight_decay"]
        epsilon = param_group["eps"]
        lr_decay = param_group["lr_decay"]
        initial_accumulator_value = param_group["initial_accumulator_value"]
        state_initializer = [make_constant_initializer(initial_accumulator_value)]
        (
            cur_rank_num_unique,
            cur_rank_unique_ids,
            unique_values,
            cur_rank_unique_embedding_grad,
        ) = self.shuffle_and_lookup(json.dumps(state_initializer))
        updated_values = flow._C.one_embedding_adagrad_update(
            cur_rank_num_unique,
            unique_values,
            cur_rank_unique_embedding_grad,
            train_step_val=step + 1,
            learning_rate_val=lr,
            scale=1.0,
            weight_decay=l2,
            lr_decay=lr_decay,
            epsilon=epsilon,
            line_size=self.storage_dim,
            embedding_size=self.embedding_dim,
            embedding_name=self.embedding_name,
        )
        flow._C.one_embedding_embedding_put(
            cur_rank_num_unique,
            cur_rank_unique_ids,
            updated_values,
            self.embedding_name,
            self.storage_dim,
        )

    def ftrl_update(self, param_group, step):
        lr = param_group["lr"]
        l2 = param_group["weight_decay"]
        lr_power = param_group["lr_power"]
        lambda1 = param_group["lambda1"]
        lambda2 = param_group["lambda2"]
        beta = param_group["beta"]
        initial_accumulator_value = param_group["initial_accumulator_value"]
        state_initializer = [
            make_constant_initializer(initial_accumulator_value),
            make_constant_initializer(initial_accumulator_value),
        ]
        (
            cur_rank_num_unique,
            cur_rank_unique_ids,
            unique_values,
            cur_rank_unique_embedding_grad,
        ) = self.shuffle_and_lookup(json.dumps(state_initializer))
        updated_values = flow._C.one_embedding_ftrl_update(
            cur_rank_num_unique,
            unique_values,
            cur_rank_unique_embedding_grad,
            learning_rate_val=lr,
            scale=1.0,
            weight_decay=l2,
            lr_power=lr_power,
            lambda1=lambda1,
            lambda2=lambda2,
            beta=beta,
            line_size=self.storage_dim,
            embedding_size=self.embedding_dim,
            embedding_name=self.embedding_name,
        )
        flow._C.one_embedding_embedding_put(
            cur_rank_num_unique,
            cur_rank_unique_ids,
            updated_values,
            self.embedding_name,
            self.storage_dim,
        )


def make_device_mem_store_options(
    persistent_path, capacity, size_factor=1, storage_dim=-1, physical_block_size=4096
):
    """make GPU only store_options param of MultiTableEmbedding

    Args:
        persistent_path (str, list): persistent storage path of Embedding. If passed a str, current rank Embedding will be saved in path/rank_id-num_ranks path. If passed a list, the list length must equals num_ranks, each elem of list represent the path of rank_id Embedding.
        capacity (int): total capacity of Embedding
        size_factor (int, optional): store size factor of embedding_dim, if SGD update, and momentum = 0, should be 1, if momentum > 0, it should be 2. if Adam, should be 3. Defaults to 1.
        storage_dim (int, optional): number of elements in embedding storage, if set storage_dim, the size_factor param will be invalid. if SGD update, and momentum = 0, storage_dim should be embedding_size*1, if momentum > 0, storage_dim should be embedding_size*2. if Adam, storage_dim should be embedding_size*3. Defaults to -1.
        physical_block_size (int, optional): physical_block_size should be sector size. Defaults to 4096.

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
        "storage_dim": storage_dim,
    }
    return options


def make_cached_ssd_store_options(
    cache_budget_mb,
    persistent_path,
    capacity=None,
    size_factor=1,
    storage_dim=-1,
    physical_block_size=4096,
    host_cache_budget_mb=0,
):
    """make SSD use GPU and host as cache store_options param of MultiTableEmbedding. If cache_budget_mb > 0 and host_cache_budget_mb > 0, use GPU and host memory as multi-level cache.

    Args:
        cache_budget_mb (int): the MB budget of per GPU as cache.
        persistent_path (str, list): persistent storage path of Embedding, must use fast SSD because of frequently random disk access during training. If passed a str, current rank Embedding will be saved in path/rank_id-num_ranks path. If passed a list, the list length must equals num_ranks, each elem of list represent the path of rank_id Embedding.
        capacity (int): total capacity of Embedding
        size_factor (int, optional): store size factor of embedding_dim, if SGD update, and momentum = 0, should be 1, if momentum > 0, it should be 2. if Adam, should be 3. Defaults to 1.
        storage_dim (int, optional): number of elements in embedding storage, if set storage_dim, the size_factor param will be invalid. if SGD update, and momentum = 0, storage_dim should be embedding_size*1, if momentum > 0, storage_dim should be embedding_size*2. if Adam, storage_dim should be embedding_size*3. Defaults to -1.
        physical_block_size (int, optional): physical_block_size should be sector size. Defaults to 4096.
        host_cache_budget_mb (int): the MB budget of host memory as cache per rank. Defaults to 0.

    Returns:
        dict: SSD use GPU and host as cache store_options param of MultiTableEmbedding

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
    assert cache_budget_mb > 0 or host_cache_budget_mb > 0
    if capacity is not None:
        assert capacity > 0
    else:
        capacity = 0

    cache_list = []
    if cache_budget_mb > 0:
        cache_list.append(
            {
                "policy": "lru",
                "cache_memory_budget_mb": cache_budget_mb,
                "value_memory_kind": "device",
            }
        )
    if host_cache_budget_mb > 0:
        cache_list.append(
            {
                "policy": "lru",
                "cache_memory_budget_mb": host_cache_budget_mb,
                "value_memory_kind": "host",
            }
        )

    options = {
        "kv_store": {
            "caches": cache_list,
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
                "capacity_hint": int(capacity),
            },
        },
        "size_factor": size_factor,
        "storage_dim": storage_dim,
    }
    return options


def make_cached_host_mem_store_options(
    cache_budget_mb,
    persistent_path,
    capacity,
    size_factor=1,
    storage_dim=-1,
    physical_block_size=4096,
):
    """make host use GPU as cache store_options param of MultiTableEmbedding

    Args:
        cache_budget_mb (int): the MB budget of per GPU as cache.
        persistent_path (str, list): persistent storage path of Embedding. If passed a str, current rank Embedding will be saved in path/rank_id-num_ranks path. If passed a list, the list length must equals num_ranks, each elem of list represent the path of rank_id Embedding.
        capacity (int): total capacity of Embedding
        size_factor (int, optional): store size factor of embedding_dim, if SGD update, and momentum = 0, should be 1, if momentum > 0, it should be 2. if Adam, should be 3. Defaults to 1.
        storage_dim (int, optional): number of elements in embedding storage, if set storage_dim, the size_factor param will be invalid. if SGD update, and momentum = 0, storage_dim should be embedding_size*1, if momentum > 0, storage_dim should be embedding_size*2. if Adam, storage_dim should be embedding_size*3. Defaults to -1.
        physical_block_size (int, optional): physical_block_size should be sector size. Defaults to 4096.

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
        "storage_dim": storage_dim,
    }
    return options


def make_uniform_initializer(low=0.0, high=1.0):
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


def make_normal_initializer(mean=0.0, std=1.0):
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


def make_constant_initializer(value):
    """make constant initializer param of make_table_options

    Args:
        constant (float): A python scalar. value to generate.

    Returns:
        dict: initializer param of make_table_options

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_constant_initializer(value=0)
        >>> # pass the initializer to flow.one_embedding.make_table_options
        >>> # ...
    """
    return {"type": "constant", "value": value}


def make_trunc_normal_initializer(mean=0.0, std=1.0, a=-2.0, b=2.0):
    """make truncated normal initializer param of make_table_options

    Args:
        mean (float): A python scalar. Mean of the random values to generate.
        std (float): A python scalar. Standard deviation of the random values to generate.
        a (float): A python scalar. The minimum cutoff value.
        b (float): A python scalar. The maximum cutoff value.

    Returns:
        dict: initializer param of make_table_options
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_trunc_normal_initializer(mean=0, std=0.01, a=-0.02, b=0.02)
        >>> # pass the initializer to flow.one_embedding.make_table_options
        >>> # ...
    """
    return {"type": "trunc_normal", "mean": mean, "std": std, "a": a, "b": b}


def make_table_options(param):
    """make table param of Embedding tables

    Args:
        param (dict or list): param can be initializer or list of column_option. initializer can be made by make_uniform_initializer or make_normal_initializer or make_constant_initializer, column options can be made by make_column_options

    Returns:
        dict: table param of Embedding tables
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        >>> table1 = flow.one_embedding.make_table_options(initializer)
        >>> table2 = flow.one_embedding.make_table_options(initializer)
        >>> tables = [table1, table2]
        >>> # pass the tables to the "tables" param of flow.one_embedding.MultiTableEmbedding or flow.one_embedding.MultiTableMultiColumnEmbedding
        >>> # ...
        
    """
    if isinstance(param, dict):
        table = {"initializer": param}
    elif isinstance(param, (list, tuple)):
        table = {"columns": param}
    else:
        raise ValueError("param must be initializer or columns")
    return table


def make_column_options(initializer):
    return {"initializer": initializer}


def make_table(param):
    """alias of `oneflow.one_embedding.make_table_options`

    See also :func:`oneflow.one_embedding.make_table_options`
    """
    return make_table_options(param)


class MultiTableEmbedding(Embedding):
    r"""MultiTableEmbedding represent multi Embedding tables with same embedding_dim, dtype, and key_type.

    Args:
        name (str): The name of Embedding
        embedding_dim (int): the size of each embedding vector
        dtype (flow.dtype): the data type of embeddings
        key_type (flow.dtype): the data type of feature ids
        tables (list): list of table param which can be made by flow.one_embedding.make_table_options
        store_options (dict): store option of Embedding
        default_initializer (dict, optional): if tables param is None, use default_initializer to initialize table. Defaults to None.
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros.
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
        padding_idx=None,
        seed=0,
    ):
        assert isinstance(embedding_dim, int)
        super().__init__(
            name,
            embedding_dim,
            dtype,
            key_type,
            tables,
            store_options,
            default_initializer,
            padding_idx,
            seed,
        )


class MultiTableMultiColumnEmbedding(Embedding):
    r"""MultiTableMultiColumnEmbedding represent multi Embedding tables with multi embedding_dim, same dtype, and key_type.

    Args:
        name (str): The name of Embedding
        embedding_dim (list): list of the size of each embedding vector
        dtype (flow.dtype): the data type of embeddings
        key_type (flow.dtype): the data type of feature ids
        tables (list): list of table param which can be made by flow.one_embedding.make_table_options
        store_options (dict): store option of Embedding
        default_initializer (dict, optional): if tables param is None, use default_initializer to initialize table. Defaults to None.
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros.
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> import oneflow.nn as nn
        >>> # a simple example with 3 table, every table has two column, the first column embedding_size is 10 and the second is 1.
        >>> # every table's first column initialize with uniform(-1/sqrt(table_size), 1/sqrt(table_size)), second column initialize with normal(0, 1/sqrt(table_size))
        >>> table_size_array = [39884407, 39043, 17289]
        >>> vocab_size = sum(table_size_array)
        >>> num_tables = len(table_size_array)
        >>> embedding_size_list = [10, 1]
        >>> scales = np.sqrt(1 / np.array(table_size_array))
        >>> tables = [
        >>>     flow.one_embedding.make_table_options(
        >>>       [flow.one_embedding.make_column_options(    
        >>>         flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)), 
        >>>        flow.one_embedding.make_column_options(    
        >>>         flow.one_embedding.make_normal_initializer(mean=0, std=scale))]
        >>>     )
        >>>     for scale in scales
        >>> ]
        >>> store_options = flow.one_embedding.make_cached_ssd_store_options(
        >>>     cache_budget_mb=8192, persistent_path="/your_path_to_ssd", capacity=vocab_size,
        >>> )
        >>> embedding = flow.one_embedding.MultiTableMultiColumnEmbedding(
        >>>     name="my_embedding",
        >>>     embedding_dim=embedding_size_list,
        >>>     dtype=flow.float,
        >>>     key_type=flow.int64,
        >>>     tables=tables,
        >>>     store_options=store_options,
        >>> )
        >>> embedding.to("cuda")
        >>> mlp = flow.nn.FusedMLP(
        >>>     in_features=sum(embedding_size_list) * num_tables,
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
        >>>         loss = self.mlp(flow.reshape(embedding, (-1, num_tables * sum(embedding_size_list))))
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
        padding_idx=None,
        seed=0,
    ):
        if isinstance(embedding_dim, (list, tuple)):
            for dim in embedding_dim:
                assert isinstance(dim, int)
        else:
            assert isinstance(embedding_dim, int)

        super().__init__(
            name,
            embedding_dim,
            dtype,
            key_type,
            tables,
            store_options,
            default_initializer,
            padding_idx,
            seed,
        )


class Ftrl(Optimizer):
    r"""FTRL Optimizer. 

    The formula is: 

        .. math:: 
                \begin{align}
                accumlator_{i+1} = accumlator_{i} + grad * grad \\
                sigma = (accumulator_{i+1}^{lr\_power} - accumulator_{i}^{lr\_power}) / learning\_rate \\
                z_{i+1} = z_{i} + grad - sigma * param_{i} \\
                \text{}
                    param_{i+1} = \begin{cases}
                    0 & \text{ if } |z_{i+1}| < \lambda_1 \\
                    -(\frac{\beta+accumlator_{i+1}^{lr\_power}}{learning\_rate} + \lambda_2)*(z_{i+1} - sign(z_{i+1})*\lambda_1) & \text{ otherwise } \\
                \end{cases}
                \end{align}
    
    Example 1: 

    .. code-block:: python 

        # Assume net is a custom model. 
        ftrl = flow.one_embedding.FTRL(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            ftrl.step()
            ftrl.zero_grad()

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate. Defaults to 1e-3.
        weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.0.
        lr_power (float, optional): learning rate decrease factor. Defaults to -0.5.
        initial_accumulator_value (float, optional): The initial value of accumlator. Defaults to 0.1.
        lambda1 (float, optional): L1 regularization strength. Defaults to 0.0.
        lambda2 (float, optional): L2 regularization strength. Defaults to 0.0.
        beta (float, optional): The value of beta. Defaults to 0.0.
    """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        weight_decay: float = 0.0,
        lr_power: float = -0.5,
        initial_accumulator_value: float = 0.1,
        lambda1: float = 0.0,
        lambda2: float = 0.0,
        beta: float = 0.0,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        options = dict()
        options["lr"] = lr
        options["weight_decay"] = weight_decay
        options["lr_power"] = lr_power
        options["initial_accumulator_value"] = initial_accumulator_value
        options["lambda1"] = lambda1
        options["lambda2"] = lambda2
        options["beta"] = beta
        super().__init__(params, options)
        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self.state[param] = dict()
                self.state[param]["accumulator_value"] = flow.zeros_like(param).fill_(
                    param_group["initial_accumulator_value"]
                )

        self._op = (
            flow.stateful_op("ftrl_update")
            .Input("model")
            .Input("model_diff")
            .Input("accumulate")
            .Input("z")
            .Build()
        )

    def step(self, closure: Callable = None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with flow.no_grad():
            loss = None
            if closure is not None:
                loss = closure()

            for param_group in self.param_groups:
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "l2": param_group["weight_decay"],
                    "lr_power": param_group["lr_power"],
                    "lambda1": param_group["lambda1"],
                    "lambda2": param_group["lambda2"],
                    "beta": param_group["beta"],
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if "z" not in self.state[param]:
                        self.state[param]["z"] = flow.zeros_like(param)

                    accumulate_tensor = self.state[param]["accumulator_value"]
                    z_tensor = self.state[param]["z"]

                    flow._C.dispatch_ftrl_update(
                        self._op,
                        (param, param.grad, accumulate_tensor, z_tensor),
                        **kwargs,
                    )

            return loss

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        new_opt_confs = []
        for param_group in self.param_groups:
            optimizer_conf = train_conf.optimizer_conf.add()

            lr = (
                param_group["initial_lr"]
                if "initial_lr" in param_group
                else param_group["lr"]
            )

            l2 = param_group["weight_decay"]
            initial_accumulator_value = param_group["initial_accumulator_value"]
            lr_power = param_group["lr_power"]
            lambda1 = param_group["lambda1"]
            lambda2 = param_group["lambda2"]
            beta = param_group["beta"]

            optimizer_conf.base_learning_rate = lr
            self._generate_lr_scale_for_optim_conf(param_group, optimizer_conf)

            optimizer_conf.ftrl_conf.initial_accumulator_value = (
                initial_accumulator_value
            )
            optimizer_conf.ftrl_conf.lr_power = lr_power
            optimizer_conf.ftrl_conf.lambda1 = lambda1
            optimizer_conf.ftrl_conf.lambda2 = lambda2
            optimizer_conf.ftrl_conf.beta = beta

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs

    @property
    def support_sparse(self):
        return False


def make_persistent_table_reader(
    paths, snapshot_name, key_type, value_type, storage_dim, physical_block_size=4096,
):
    r"""Creates a reader for reading persistent table.

    Args:
        paths (list): paths of tables to read
        snapshot_name (str): name of the snapshot to read
        key_type (flow.dtype): the data type of key
        value_type (flow.dtype): the data type of value
        storage_dim (int): number of elements in each value
        physical_block_size (int, optional): physical_block_size should be sector size. Defaults to 4096
    """
    return PersistentTableReader(
        paths,
        snapshot_name,
        key_type,
        value_type,
        storage_dim,
        4 * 1024,
        physical_block_size,
    )


def make_persistent_table_writer(
    paths, snapshot_name, key_type, value_type, storage_dim, physical_block_size=4096,
):
    r"""Creates a writer for writing persistent table.

    Args:
        paths (list): paths of tables to write
        snapshot_name (str): name of the snapshot to write
        key_type (flow.dtype): the data type of key
        value_type (flow.dtype): the data type of value
        storage_dim (int): number of elements in each value
        physical_block_size (int, optional): physical_block_size should be sector size. Defaults to 4096
    """
    return PersistentTableWriter(
        paths,
        snapshot_name,
        key_type,
        value_type,
        storage_dim,
        4 * 1024,
        physical_block_size,
    )


class SmartDecayAdam(flow.nn.optimizer.adam.Adam):
    """Implements SmartDecayAdam algorithm.
       The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
       For Sparse Embedding Table in OneEmbedding, implement the SmartDecayAdam algorithm.
       For other models, it is same as Adam.
    """

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        new_opt_confs = super()._generate_conf_for_graph(train_conf, vars_conf)
        for opt_conf in new_opt_confs:
            opt_conf.adam_conf.smart_decay = True


class Optimizer(Optimizer):
    def __init__(
        self, optimizer: Optimizer, embeddings: List[Embedding],
    ):
        self.optimizer = optimizer
        self.embeddings = embeddings
        self.param_groups = optimizer.param_groups
        # self._default_options = optimizer._default_options
        # self._state = optimizer._state
        self.defaults = optimizer.defaults
        self.state = optimizer.state
        self.embedding_param_group_dict = {}
        for embedding in self.embeddings:
            for group in self.param_groups:
                param_set = set()
                for param in group.parameters:
                    param_set.add(param)
                if embedding.shadow in param_set:
                    self.embedding_param_group_dict[embedding.embedding_name] = group
            if not embedding.embedding_name in self.embedding_param_group_dict:
                raise ValueError("embedding must in optimizers param_group")

    def step(self, closure: Callable = None):
        step = self.optimizer.state["step"]
        for embedding in self.embeddings:
            param_group = self.embedding_param_group_dict[embedding.embedding_name]
            if type(self.optimizer) is flow.nn.optimizer.sgd.SGD:
                embedding.sgd_update(param_group, step)
            elif type(self.optimizer) is flow.nn.optimizer.adam.Adam:
                embedding.adam_update(param_group, step)
            elif type(self.optimizer) is flow.nn.optimizer.adagrad.Adagrad:
                embedding.adagrad_update(param_group, step)
            elif type(self.optimizer) is flow.one_embedding.Ftrl:
                embedding.ftrl_update(param_group, step)
            else:
                raise NotImplementedError("only support sgd, adam, adagrad and ftrl")
        self.optimizer.step()

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        return self.optimizer._generate_conf_for_graph(train_conf, vars_conf)
