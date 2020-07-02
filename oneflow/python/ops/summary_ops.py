from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_op_builder as user_op_builder

import oneflow as flow


@oneflow_export("summary.scalar")
def write_scalar(value, step, tag, name=None):
    if name is None:
        name = id_util.UniqueStr("WriteScalar_")
    (
        flow.user_op_builder(name)
        .Op("write_scalar")
        .Input("in", [value])
        .Input("step", [step])
        .Input("tag", [tag])
        .Build()
        .InferAndTryRun()
    )


@oneflow_export("summary.create_summary_writer")
def create_summary_write(logdir, name=None):
    if name is None:
        name = id_util.UniqueStr("CreateWriter_")
    (
        flow.user_op_builder(name)
        .Op("create_summary_writer")
        .Attr("logdir", logdir, "AttrTypeString")
        .Build()
        .InferAndTryRun()
    )


@oneflow_export("summary.histogram")
def write_histogram(value, step, tag, name=None):
    if name is None:
        name = id_util.UniqueStr("WriteHistogram_")
    (
        flow.user_op_builder(name)
        .Op("write_histogram")
        .Input("in", [value])
        .Input("step", [step])
        .Input("tag", [tag])
        .Build()
        .InferAndTryRun()
    )


@oneflow_export("summary.text")
def write_text(value, step, tag, name=None):
    if name is None:
        name = id_util.UniqueStr("WriteText_")
    (
        flow.user_op_builder(name)
        .Op("write_text")
        .Input("in", [value])
        .Input("step", [step])
        .Input("tag", [tag])
        .Build()
        .InferAndTryRun()
    )


@oneflow_export("summary.pb")
def write_pb(value, step=None, name=None):
    if name is None:
        name = id_util.UniqueStr("WritePb_")
    (
        flow.user_op_builder(name)
        .Op("write_pb")
        .Input("in", [value])
        .Input("step", [step])
        .Build()
        .InferAndTryRun()
    )


@oneflow_export("summary.image")
# def write_image(data, step=None, bad_color=None, tag=None, max_images=None, name=None):
def write_image(value, step=None, tag=None, name=None):
    if name is None:
        name = id_util.UniqueStr("WriteImage_")
    if tag is None:
        tag = "image"
    (
        flow.user_op_builder(name)
        .Op("write_image")
        .Input("in", [value])
        .Input("step", [step])
        .Input("tag", [tag])
        #        .Input("max_images", [max_images])
        #        .Input("bad_color", [bad_color])
        .Build()
        .InferAndTryRun()
    )
