from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_op_builder as user_op_builder

import oneflow as flow


# @oneflow_export("summary.scalar")
# def write_scalar(value, step=None, tag=None, name=None):
#     if name is None:
#         name = id_util.UniqueStr("WriteScalar_")
#     if tag is None:
#         tag = "scalar"
#     (flow.user_op_builder(name).Op("write_scalar")
#     .Input("in", [value])
#     .Attr("step", step, "AttrTypeInt64")
#     .Attr("tag", tag, "AttrTypeString")
#     .Build()
#     .InferAndTryRun())


@oneflow_export("summary.scalar")
def write_scalar(value, step=None, tag=None, name=None):
    if name is None:
        name = id_util.UniqueStr("WriteScalar_")
    if tag is None:
        tag = "scalar"
    (flow.user_op_builder(name).Op("write_scalar")
    .Input("in", [value])
    .Input("step", [step])
    .Input("tag", [tag])
    .Build()
    .InferAndTryRun())

@oneflow_export("summary.create_summary_writer")
def create_summary_write(logdir, name=None):
    if name is None:
        name = id_util.UniqueStr("CreateWriter_")
    (flow.user_op_builder(name).Op("create_summary_writer")
    .Attr("logdir", logdir, "AttrTypeString")
    .Build()
    .InferAndTryRun())
