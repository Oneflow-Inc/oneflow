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


class Embedding(Module):
    def __init__(
        self,
        name,
        embedding_dim,
        dtype,
        key_type,
        columns,
        store_options,
        default_initializer={"type": "normal", "mean": 0, "std": 0.05},
    ):
        super().__init__()
        key_value_store_options = {}
        embedding_columns = {}
        key_value_store_options["name"] = name
        assert embedding_dim > 0
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.key_type = key_type

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
        assert kv_store.__contains__("persistent_table")
        persistent_table = kv_store["persistent_table"]
        assert isinstance(persistent_table, dict)
        assert persistent_table.__contains__("path")
        persistent_table_path = persistent_table["path"]
        assert isinstance(persistent_table_path, (str, list, tuple))
        if isinstance(persistent_table_path, (list, tuple)):
            assert len(persistent_table_path) == flow.env.get_world_size()
        if persistent_table.__contains__("physical_block_size"):
            assert persistent_table["physical_block_size"] in [512, 4096]
        else:
            persistent_table["physical_block_size"] = 4096
        key_value_store_options["kv_store"] = kv_store

        # initializer
        if columns is not None:
            assert isinstance(columns, (list, tuple))
            for column in columns:
                assert isinstance(column, dict)
                assert column.__contains__("initializer")
                _check_initializer(column["initializer"])
            embedding_columns["columns"] = columns
        else:
            assert default_initializer is not None
            _check_initializer(default_initializer)
            embedding_columns["columns"] = [{"initializer": default_initializer}]
        key_value_store_options["parallel_num"] = flow.env.get_world_size()
        self.key_value_store_options = json.dumps(key_value_store_options)
        self.embedding_columns = json.dumps(embedding_columns)
        self.num_columns = len(embedding_columns["columns"])
        self.local_rank = flow.env.get_local_rank()
        self.rank_id = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
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
        self.handler.SaveSnapshot(snapshot_name)

    def load_snapshot(self, snapshot_name):
        self.handler.LoadSnapshot(snapshot_name)

    def forward(self, ids, column_ids=None):
        assert self.key_type == ids.dtype, "ids data_type must equals key_type"
        return flow._C.one_embedding_lookup(
            self.shadow,
            ids,
            column_ids,
            self.dtype,
            self.embedding_dim,
            self.num_columns,
            self.embedding_columns,
            self.key_value_store_options,
        )


def _check_and_get_capacity_and_memory_budget_mb(
    capacity, capacity_name, memory_budget_mb, memory_budget_mb_name
):
    if capacity is not None:
        if memory_budget_mb is not None:
            print(
                "WARNING: {} is set so {} will be ignored".format(
                    capacity_name, memory_budget_mb_name
                )
            )
            print(traceback.format_stack()[-3])
        assert capacity > 0
        memory_budget_mb = 0
    elif memory_budget_mb is not None:
        assert memory_budget_mb > 0
        capacity = 0
    else:
        raise ValueError(
            "must set {} or {}".format(capacity_name, memory_budget_mb_name)
        )
    return capacity, memory_budget_mb


def make_device_mem_store_options(
    persistent_path,
    capacity_per_rank=None,
    device_memory_budget_mb_per_rank=None,
    size_factor=1,
    physical_block_size=512,
):
    assert isinstance(persistent_path, (str, list, tuple))
    (
        capacity_per_rank,
        device_memory_budget_mb_per_rank,
    ) = _check_and_get_capacity_and_memory_budget_mb(
        capacity_per_rank,
        "capacity_per_rank",
        device_memory_budget_mb_per_rank,
        "device_memory_budget_mb_per_rank",
    )
    assert capacity_per_rank > 0 or device_memory_budget_mb_per_rank > 0
    options = {
        "kv_store": {
            "caches": [
                {
                    "policy": "full",
                    "capacity": capacity_per_rank,
                    "cache_memory_budget_mb": device_memory_budget_mb_per_rank,
                    "value_memory_kind": "device",
                }
            ],
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
            },
        },
        "size_factor": size_factor,
    }
    return options


def make_host_mem_store_options(
    persistent_path,
    capacity_per_rank=None,
    host_memory_budget_mb_per_rank=None,
    size_factor=1,
    physical_block_size=512,
):
    assert isinstance(persistent_path, (str, list, tuple))
    (
        capacity_per_rank,
        host_memory_budget_mb_per_rank,
    ) = _check_and_get_capacity_and_memory_budget_mb(
        capacity_per_rank,
        "capacity_per_rank",
        host_memory_budget_mb_per_rank,
        "host_memory_budget_mb_per_rank",
    )
    assert capacity_per_rank > 0 or host_memory_budget_mb_per_rank > 0
    options = {
        "kv_store": {
            "caches": [
                {
                    "policy": "full",
                    "capacity": capacity_per_rank,
                    "cache_memory_budget_mb": host_memory_budget_mb_per_rank,
                    "value_memory_kind": "host",
                }
            ],
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
            },
        },
        "size_factor": size_factor,
    }
    return options


def make_device_mem_cached_ssd_store_options(
    persistent_path,
    cached_capacity_per_rank=None,
    device_memory_budget_mb_per_rank=None,
    size_factor=1,
    physical_block_size=512,
):
    assert isinstance(persistent_path, (str, list, tuple))
    (
        cached_capacity_per_rank,
        device_memory_budget_mb_per_rank,
    ) = _check_and_get_capacity_and_memory_budget_mb(
        cached_capacity_per_rank,
        "cached_capacity_per_rank",
        device_memory_budget_mb_per_rank,
        "device_memory_budget_mb_per_rank",
    )
    assert cached_capacity_per_rank > 0 or device_memory_budget_mb_per_rank > 0
    options = {
        "kv_store": {
            "caches": [
                {
                    "policy": "lru",
                    "capacity": cached_capacity_per_rank,
                    "cache_memory_budget_mb": device_memory_budget_mb_per_rank,
                    "value_memory_kind": "device",
                }
            ],
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
            },
        },
        "size_factor": size_factor,
    }
    return options


def make_host_mem_cached_ssd_store_options(
    persistent_path,
    cached_capacity_per_rank=None,
    host_memory_budget_mb_per_rank=None,
    size_factor=1,
    physical_block_size=512,
):
    assert isinstance(persistent_path, (str, list, tuple))
    (
        cached_capacity_per_rank,
        host_memory_budget_mb_per_rank,
    ) = _check_and_get_capacity_and_memory_budget_mb(
        cached_capacity_per_rank,
        "cached_capacity_per_rank",
        host_memory_budget_mb_per_rank,
        "host_memory_budget_mb_per_rank",
    )
    assert cached_capacity_per_rank > 0 or host_memory_budget_mb_per_rank > 0
    options = {
        "kv_store": {
            "caches": [
                {
                    "policy": "lru",
                    "capacity": cached_capacity_per_rank,
                    "cache_memory_budget_mb": host_memory_budget_mb_per_rank,
                    "value_memory_kind": "host",
                }
            ],
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
            },
        },
        "size_factor": size_factor,
    }
    return options


def make_device_mem_cached_host_mem_store_options(
    persistent_path,
    cached_capacity_per_rank=None,
    device_memory_budget_mb_per_rank=None,
    capacity_per_rank=None,
    host_memory_budget_mb_per_rank=None,
    size_factor=1,
    physical_block_size=512,
):
    assert isinstance(persistent_path, (str, list, tuple))
    (
        cached_capacity_per_rank,
        device_memory_budget_mb_per_rank,
    ) = _check_and_get_capacity_and_memory_budget_mb(
        cached_capacity_per_rank,
        "cached_capacity_per_rank",
        device_memory_budget_mb_per_rank,
        "device_memory_budget_mb_per_rank",
    )
    assert cached_capacity_per_rank > 0 or device_memory_budget_mb_per_rank > 0
    (
        capacity_per_rank,
        host_memory_budget_mb_per_rank,
    ) = _check_and_get_capacity_and_memory_budget_mb(
        capacity_per_rank,
        "capacity_per_rank",
        host_memory_budget_mb_per_rank,
        "host_memory_budget_mb_per_rank",
    )
    assert capacity_per_rank > 0 or host_memory_budget_mb_per_rank > 0
    options = {
        "kv_store": {
            "caches": [
                {
                    "policy": "lru",
                    "capacity": cached_capacity_per_rank,
                    "cache_memory_budget_mb": device_memory_budget_mb_per_rank,
                    "value_memory_kind": "device",
                },
                {
                    "policy": "full",
                    "capacity": capacity_per_rank,
                    "cache_memory_budget_mb": host_memory_budget_mb_per_rank,
                    "value_memory_kind": "host",
                },
            ],
            "persistent_table": {
                "path": persistent_path,
                "physical_block_size": physical_block_size,
            },
        },
        "size_factor": size_factor,
    }
    return options
