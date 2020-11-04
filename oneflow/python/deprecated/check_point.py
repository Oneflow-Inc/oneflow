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
import datetime
import os
import shutil

import numpy as np
from google.protobuf import text_format

import oneflow
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.ops.initializer_util as initializer_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.hob as hob
import oneflow.python.framework.job_instance as job_instance
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.op_arg_util as op_arg_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.lib.core.async_util as async_util
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.op_executor as op_executor
import oneflow.python.eager.op_infer_util as op_infer_util
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
import oneflow.core.framework.user_op_attr_pb2 as attr_value_pb
from oneflow.python.experimental import interface_op_read_and_write
from oneflow.python.framework.remote_blob import EagerBlobTrait
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.ops.get_variable as get_variable

from oneflow.python.oneflow_export import oneflow_export
from typing import Dict, List, Union, Sequence, Optional


@oneflow_export("train.CheckPoint")
class CheckPoint(object):
    """Create a `CheckPoint` object to manage checkpoint manually.

    """

    def __init__(self) -> None:
        pass

    @session_ctx.try_init_default_session
    def save(self, path: str) -> None:
        r"""save a checkpoint to `path`.

        Args:
            path: A `string` of path to save checkpoint. 
        """
        if not config_util.api_legacy_model_io_enabled():
            print(
                "'checkpoint.save()' is deprecated. Please use the new checkpoint API"
            )
            Save(GetAllVariables(), path)
            return
        assert type(path) is str
        enable_if.unique([lazy_checkpoint_save, eager_checkpoint_save])(path)

    @session_ctx.try_init_default_session
    def init(self) -> None:
        r"""Initialize models by default initializer of op or Job.
        """
        if not config_util.api_legacy_model_io_enabled():
            print(
                "'checkpoint.init()' is deprecated. It has no effect and will be removed in the future"
            )
            return
        enable_if.unique([lazy_checkpoint_init, eager_checkpoint_init])()

    @session_ctx.try_init_default_session
    def load(self, path: str) -> None:
        r"""load a checkpoint from `path` and initialize models.

        Args:
            path: A `string` of path to load checkpoint.
        """
        if not config_util.api_legacy_model_io_enabled():
            print(
                "'checkpoint.load()' is deprecated. Please use the new checkpoint API"
            )
            LoadVariables(Load(path))
            return
        assert type(path) is str
        enable_if.unique([lazy_checkpoint_load, eager_checkpoint_load])(path)


@enable_if.condition(hob.in_normal_mode & ~hob.eager_execution_enabled)
def lazy_checkpoint_save(path):
    session_ctx.GetDefaultSession().LaunchJob(_MakeModelSaveJobFunc(path))


@enable_if.condition(hob.in_normal_mode & ~hob.eager_execution_enabled)
def lazy_checkpoint_init():
    session_ctx.GetDefaultSession().LaunchJob(_MakeModelInitJobFunc())


@enable_if.condition(hob.in_normal_mode & ~hob.eager_execution_enabled)
def lazy_checkpoint_load(path):
    session_ctx.GetDefaultSession().LaunchJob(_MakeModelLoadJobFunc(path))


@enable_if.condition(hob.in_normal_mode & hob.eager_execution_enabled)
def eager_checkpoint_save(path):
    op_executor.EagerSaveVariableBlob(path)


@enable_if.condition(hob.in_normal_mode & hob.eager_execution_enabled)
def eager_checkpoint_init():
    # eager variables are initialized in oneflow.get_variable()
    pass


@enable_if.condition(hob.in_normal_mode & hob.eager_execution_enabled)
def eager_checkpoint_load(path):
    session_ctx.GetDefaultSession().snapshot_mgr.load(path)


def _MakeModelInitJobFunc():
    def push_cb(blob):
        pass

    def finish_cb():
        pass

    sess = session_ctx.GetDefaultSession()
    return job_instance.MakeJobInstance(
        str(sess.inter_user_job_info.global_model_init_job_name),
        push_cb=push_cb,
        finish_cb=finish_cb,
    )


def _MakeModelLoadJobFunc(path):
    def push_cb(blob):
        blob.CopyFromNdarray(np.frombuffer(path.encode("ascii"), dtype=np.int8))

    def finish_cb():
        pass

    sess = session_ctx.GetDefaultSession()
    return job_instance.MakeJobInstance(
        str(sess.inter_user_job_info.global_model_load_job_name),
        push_cb=push_cb,
        finish_cb=finish_cb,
    )


def _MakeModelSaveJobFunc(path):
    def push_cb(blob):
        blob.CopyFromNdarray(np.frombuffer(path.encode("ascii"), dtype=np.int8))

    def finish_cb():
        pass

    sess = session_ctx.GetDefaultSession()
    return job_instance.MakeJobInstance(
        str(sess.inter_user_job_info.global_model_save_job_name),
        push_cb=push_cb,
        finish_cb=finish_cb,
    )
