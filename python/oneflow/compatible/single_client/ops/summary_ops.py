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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework import id_util as id_util
from oneflow.compatible.single_client.ops import user_op_builder as user_op_builder


def write_scalar(value, step, tag, name=None):
    """Write scalar to log file

    Args:
        value: A 'Blob' with 1 value and dtype in (flow.float, flow.double, flow.int64, flow.int32)
        step: A 'Blob' with 1 value and dtype is 'flow.int64'
        tag: A 'Blob' with 1 value and dtype is 'flow.int8'
        name: This operator's name 
    """
    if name is None:
        name = id_util.UniqueStr("WriteScalar_")
    flow.user_op_builder(name).Op("summary_write_scalar").Input("in", [value]).Input(
        "step", [step]
    ).Input("tag", [tag]).Build().InferAndTryRun()


def create_summary_writer(logdir, name=None):
    """Create a summary writer object

    Args:
        logdir: log dir
        name: This operator's name
    """
    if name is None:
        name = id_util.UniqueStr("CreateWriter_")
    flow.user_op_builder(name).Op("create_summary_writer").Attr(
        "logdir", logdir
    ).Build().InferAndTryRun()


def flush_summary_writer(name=None):
    """Flush the summary writer

    Args:
        name: This operator's name
    """
    if name is None:
        name = id_util.UniqueStr("FlushWriter_")
    flow.user_op_builder(name).Op("flush_summary_writer").Build().InferAndTryRun()


def write_histogram(value, step, tag, name=None):
    """Write histogram to log file

    Args:
        value: A 'Blob' with dtype in (flow.float, flow.double, flow.int64, flow.int32, flow.int8, flow.uint8)
        step: A 'Blob' with 1 value and dtype is 'flow.int64'
        tag: A 'Blob' with 1 value and dtype is 'flow.int8'
        name: This operator's name 
    """
    if name is None:
        name = id_util.UniqueStr("WriteHistogram_")
    flow.user_op_builder(name).Op("summary_write_histogram").Input("in", [value]).Input(
        "step", [step]
    ).Input("tag", [tag]).Build().InferAndTryRun()


def write_pb(value, step=None, name=None):
    """Write raw protobuf data to log file

    Args:
        value: A 'Blob' with dtype in 'flow.int8'
        step: A 'Blob' with 1 value and dtype is 'flow.int64'
        name: This operator's name 
    """
    if name is None:
        name = id_util.UniqueStr("WritePb_")
    flow.user_op_builder(name).Op("summary_write_pb").Input("in", [value]).Input(
        "step", [step]
    ).Build().InferAndTryRun()


def write_image(value, step=None, tag=None, name=None):
    """Write image to log file

    Args:
        value: A 'Blob' with dtype in 'flow.uint8'
        step: A 'Blob' with 1 value and dtype is 'flow.int64'
        tag: A 'Blob' with 1 value and dtype is 'flow.int8'
        name: This operator's name 
    """
    if name is None:
        name = id_util.UniqueStr("WriteImage_")
    if tag is None:
        tag = "image"
    flow.user_op_builder(name).Op("summary_write_image").Input("in", [value]).Input(
        "step", [step]
    ).Input("tag", [tag]).Build().InferAndTryRun()
