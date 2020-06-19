from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.python.test.customized.hparams as hp

import oneflow as flow


@oneflow_export("summary.scalar")
def write_scalar(value, step, tag, name=None):
    if name is None:
        name = id_util.UniqueStr("WriteScalar_")
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


@oneflow_export("summary.histogram")
def write_hsitogram(value, step, tag, name=None):
    if name is None:
        name = id_util.UniqueStr("WriterHistogram_")
    (flow.user_op_builder(name).Op("write_histogram")
     .Input("in", [value])
     .Input("step", [step])
     .Input("tag", [tag])
     .Build()
     .InferAndTryRun())


@oneflow_export("summary.text")
def write_hsitogram(value, step, tag, name=None):
    if name is None:
        name = id_util.UniqueStr("WriterText_")
    (flow.user_op_builder(name).Op("write_text")
     .Input("in", [value])
     .Input("step", [step])
     .Input("tag", [tag])
     .Build()
     .InferAndTryRun())


@oneflow_export("summary.pb")
def write_pb(value, step=None, name=None):
    if name is None:
        name = id_util.UniqueStr("WritePb_")
    (flow.user_op_builder(name).Op("write_pb")
     .Input("in", [value])
     .Input("step", [step])
     .Build()
     .InferAndTryRun())


@oneflow_export("summary.hparam")
def write_hparam(value=None, step=None, tag=None, name=None):
    if name is None:
        name = id_util.UniqueStr("WriteHparam_")
    hparams = {
        hp.HParam("learning_rate", hp.RealInterval(1e-2, 1e-1)): 0.02,
        hp.HParam("dense_layers", hp.IntInterval(2, 7)): 5,
        hp.HParam("optimizer", hp.Discrete(["adam", "sgd"])): "adam",
        hp.HParam("who_knows_what"): "???",
        hp.HParam(
            "magic",
            hp.Discrete([False, True]),
            display_name="~*~ Magic ~*~",
            description="descriptive",
        ): True,
        "dropout": 0.3,
    }
    normalized_hparams = {
        "learning_rate": 0.02,
        "dense_layers": 5,
        "optimizer": "adam",
        "who_knows_what": "???",
        "magic": True,
        "dropout": 0.3,
    }
    start_time_secs = 123.45
    trial_id = "psl27"

    hp.hparams(hparams, "psl27", 123.45)

