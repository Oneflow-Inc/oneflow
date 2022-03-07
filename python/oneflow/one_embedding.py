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
import warnings
from typing import Optional

import oneflow as flow
import oneflow._oneflow_internal
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _check_inplace_valid
import json
import os
import datetime
from oneflow._oneflow_internal import OneEmbeddingHandler
import numpy as np


def _check_initializer(initializer):
    assert isinstance(initializer, dict)
    assert initializer.__contains__("type")
    initializer_type = initializer["type"]
    assert initializer_type in ["uniform", "normal"]
    if initializer_type == "uniform":
        assert initializer.__contains__("low")
        assert initializer.__contains__("high")
    else:
        assert initializer.__contains__("mean")
        assert initializer.__contains__("std")


def _check_cache(cache):
    assert isinstance(cache, dict)
    assert cache.__contains__("policy")
    assert cache["policy"] in ["lru", "full"]
    assert cache.__contains__("cache_memory_budget_mb")
    assert cache["cache_memory_budget_mb"] > 0
    assert cache.__contains__("value_memory_kind")
    assert cache["value_memory_kind"] in ["device", "host"]


class Embedding(Module):
    def __init__(self, name, embedding_dim, dtype, key_type, intializer, store_options):
        super().__init__()
        embedding_options = {}
        embedding_options["name"] = name
        assert embedding_dim > 0
        embedding_options["embedding_dim"] = embedding_dim
        self.dtype = dtype
        self.key_type = key_type

        scale_factor = store_options["size_factor"]
        embedding_options["storage_dim"] = scale_factor * embedding_dim

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
        if isinstance(persistent_table_path, str):
            assert os.path.exists(persistent_table_path)
        else:
            assert len(persistent_table_path) == flow.env.get_world_size()
            for i in range(len(persistent_table_path)):
                assert os.path.exists(persistent_table_path[i])
        if persistent_table.__contains__("physical_block_size"):
            assert persistent_table["physical_block_size"] in [512, 4096]
        else:
            persistent_table["physical_block_size"] = 4096
        embedding_options["kv_store"] = kv_store

        # initializer
        assert intializer.__contains__("default_initializer")
        _check_initializer(intializer["default_initializer"])
        if intializer.__contains__("columns"):
            columns = intializer["columns"]
            assert isinstance(columns, (list, tuple))
            for column in columns:
                assert isinstance(column, dict)
                assert column.__contains__("initializer")
                _check_initializer(column["initializer"])
        embedding_options["default_initializer"] = intializer["default_initializer"]
        embedding_options["columns"] = intializer["columns"]

        self.embedding_options = json.dumps(embedding_options)
        # TODO(zzk): Support placement configuration. Currently OneEmbedding is placed in all gpu.
        self.parallel_id = flow.env.get_rank()
        self.parallel_num = flow.env.get_world_size()
        self.handler = OneEmbeddingHandler(
            self.embedding_options, self.parallel_id, self.parallel_num
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

    def forward(self, ids, column_ids):
        assert self.key_type == ids.dtype, "ids data_type must equals key_type"
        return flow._C.embedding_lookup_placeholder(
            self.shadow, ids, column_ids, self.dtype, self.embedding_options,
        )


def make_device_mem_store_option(
    device_memory_mb, persistent_path, size_factor=1, physical_block_size=512
):
    option = {
        "kv_store": {
            "caches": [
                {
                    "policy": "full",
                    "cache_memory_budget_mb": device_memory_mb,
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
    return option


def make_host_mem_store_option(
    host_memory_mb, persistent_path, size_factor=1, physical_block_size=512
):
    option = {
        "kv_store": {
            "caches": [
                {
                    "policy": "full",
                    "cache_memory_budget_mb": host_memory_mb,
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
    return option


def make_device_mem_cached_ssd_store_option(
    device_memory_mb, persistent_path, size_factor=1, physical_block_size=512
):
    option = {
        "kv_store": {
            "caches": [
                {
                    "policy": "lru",
                    "cache_memory_budget_mb": device_memory_mb,
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
    return option


def make_host_mem_cached_ssd_store_option(
    host_memory_mb, persistent_path, size_factor=1, physical_block_size=512
):
    option = {
        "kv_store": {
            "caches": [
                {
                    "policy": "lru",
                    "cache_memory_budget_mb": host_memory_mb,
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
    return option


def make_device_mem_cached_host_store_option(
    device_memory_mb,
    host_memory_mb,
    persistent_path,
    size_factor=1,
    physical_block_size=512,
):
    option = {
        "kv_store": {
            "caches": [
                {
                    "policy": "lru",
                    "cache_memory_budget_mb": device_memory_mb,
                    "value_memory_kind": "device",
                },
                {
                    "policy": "full",
                    "cache_memory_budget_mb": host_memory_mb,
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
    return option
