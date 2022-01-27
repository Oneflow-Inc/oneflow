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


class OneEmbeddingLookup(Module):
    def __init__(self, embedding_options):
        super().__init__()
        # self.dtype = embedding_options["dtype"]
        assert embedding_options.__contains__("name")
        self.emb_name = embedding_options["name"]
        assert embedding_options.__contains__("embedding_dim")
        embedding_dim = embedding_options["embedding_dim"]
        assert embedding_dim > 0
        if embedding_options.__contains__("cache"):
            cache = embedding_options["cache"]
            assert isinstance(cache, (dict, list, tuple))
            if isinstance(cache, dict):
                _check_cache(cache)
                cache = [cache]
            else:
                assert len(cache) <= 2
                for i in range(len(cache)):
                    assert isinstance(cache[i], dict)
                    _check_cache(cache[i])

        # kv store
        assert embedding_options.__contains__("kv_store")
        kv_store = embedding_options["kv_store"]
        assert isinstance(kv_store, dict)
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

        # initializer
        assert embedding_options.__contains__("default_initializer")
        _check_initializer(embedding_options["default_initializer"])
        if embedding_options.__contains__("columns"):
            columns = embedding_options["columns"]
            assert isinstance(columns, (list, tuple))
            for column in columns:
                assert isinstance(column, dict)
                assert column.__contains__("initializer")
                _check_initializer(column["initializer"])

        # optimizer
        assert embedding_options.__contains__("optimizer")
        optimizer = embedding_options["optimizer"]
        assert isinstance(optimizer, dict)
        assert optimizer.__contains__("type")
        optimizer_type = optimizer["type"]
        assert optimizer_type in ["sgd", "adam"]
        if optimizer_type == "sgd":
            if optimizer.__contains__("momentum") and optimizer["momentum"] > 0:
                optimizer["type"] = "momentum"
            else:
                optimizer["momentum"] = 0.0
        elif optimizer_type == "adam":
            if optimizer.__contains__("betas"):
                assert isinstance(optimizer["betas"], (list, tuple))
            else:
                optimizer["betas"] = [0.9, 0.999]
            if optimizer.__contains__("eps"):
                assert optimizer["eps"] >= 0 and optimizer["eps"] < 1
            else:
                optimizer["eps"] = 1e-8
            if optimizer.__contains__("do_bias_correction"):
                assert isinstance(optimizer["do_bias_correction"], bool)
            else:
                optimizer["do_bias_correction"] = True
        else:
            raise NotImplementedError("only support sgd and adam")

        assert optimizer.__contains__("lr")
        lr_schedule = optimizer["lr"]
        assert lr_schedule.__contains__("base_lr")
        if lr_schedule.__contains__("decay"):
            decay = lr_schedule["decay"]
            assert isinstance(decay, dict)
            assert decay.__contains__("type")
            decay_type = decay["type"]
            assert decay_type in [
                "polynomial",
            ]
            if decay_type == "polynomial":
                assert decay.__contains__("decay_batches")
                if not decay.__contains__("end_lr"):
                    decay["end_lr"] = 0.0
                if not decay.__contains__("power"):
                    decay["power"] = 2.0
                if not decay.__contains__("cycle"):
                    decay["cycle"] = False
            else:
                raise NotImplementedError("unsupported decay type")
        if lr_schedule.__contains__("warmup"):
            warmup = lr_schedule["warmup"]
            assert isinstance(warmup, dict)
            assert warmup.__contains__("type")
            warmup_type = warmup["type"]
            assert warmup_type in ["linear", "constant"]
            if warmup_type == "linear":
                assert warmup.__contains__("warmup_batches")
                assert warmup.__contains__("start_multiplier")
            elif warmup_type == "constant":
                assert warmup.__contains__("warmup_batches")
                assert warmup.__contains__("multiplier")
            else:
                raise NotImplementedError("unsupported warmup type")
        self.dtype = embedding_options["dtype"]
        del embedding_options["dtype"]
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
        return flow._C.embedding_lookup_placeholder(
            self.shadow, ids, column_ids, self.dtype, self.embedding_options,
        )
