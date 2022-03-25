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
        self.handler.SaveSnapshot(snapshot_name)

    def load_snapshot(self, snapshot_name):
        self.handler.LoadSnapshot(snapshot_name)

    def forward(self, ids, table_ids=None):
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
    return {"type": "uniform", "low": low, "high": high}


def make_normal_initializer(mean, std):
    return {"type": "normal", "mean": mean, "std": std}


def make_table(initializer):
    return {"initializer": initializer}
