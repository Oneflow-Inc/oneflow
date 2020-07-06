import numpy as np

import oneflow.customized.utils.projector_pb2 as projector_pb2
from oneflow.python.oneflow_export import oneflow_export
import time

import oneflow as flow


def _write_summary_projector(filename=None, summary_projector=None):
    with open(filename, "wb") as f:
        f.write(summary_projector.SerializeToString())
        f.flush()


@oneflow_export("embedding_projector")
def embedding_projector(
    value=None,
    label=None,
    tag=None,
    step=None,
    logdir=None,
    sample_name=None,
    sample_type=None,
    x=None,
):
    if tag is None:
        tag = "embedding_projector"
    summary_projector = projector_pb2.SummaryProjector()
    summary_projector.metadata.type = projector_pb2.MetaData.ProjectorType.EMBEDDING
    projector = summary_projector.projector.add()
    _set_projector(pro=projector, tag=tag, step=step, value=value, label=label)
    if (sample_name is not None) and (sample_type is not None):
        _set_sample(
            sample=summary_projector.sample,
            name=sample_name,
            x=x,
            sample_type=sample_type,
        )
    embedding_prefix = "/projector."
    suffix = ".v2"
    filename = logdir + embedding_prefix + str(time.time()) + suffix
    _write_summary_projector(filename=filename, summary_projector=summary_projector)


@oneflow_export("exception_projector")
def exception_projector(
    value=None,
    tag=None,
    step=None,
    logdir=None,
    sample_name=None,
    sample_type=None,
    x=None,
):
    if tag is None:
        tag = "exception_projector"
    summary_projector = projector_pb2.SummaryProjector()
    summary_projector.metadata.type = projector_pb2.MetaData.ProjectorType.EXCEPTION
    projector = summary_projector.projector.add()
    _set_projector(pro=projector, tag=tag, step=step, value=value)
    if (sample_name is not None) and (sample_type is not None):
        _set_sample(
            sample=summary_projector.sample,
            name=sample_name,
            x=x,
            sample_type=sample_type,
        )
    exception_prefix = "/projector.gradient."
    suffix = ".v2"
    filename = logdir + exception_prefix + str(time.time()) + suffix
    _write_summary_projector(filename=filename, summary_projector=summary_projector)


def _set_tensor(tensor: projector_pb2.Tensor, value):
    for d in value.shape:
        td = tensor.shape.dim.add()
        td.size = d
    tensor.dtype = str(value.dtype)
    tensor.content = value.tobytes()
    return


def _set_projector(pro, tag, step, value, label=None):
    pro.tag = str(tag)
    pro.step = step
    pro.WALL_TIME = time.time()
    _set_tensor(pro.value, value)
    if label is not None:
        _set_tensor(pro.label, label)
    return


def _set_sample(sample, name, x, sample_type):
    if name is not None:
        sample.name = name
    if sample_type == "image" or sample_type == "IMAGE":
        sample.type = projector_pb2.Sample.SampleType.IMAGE
    elif sample_type == "audio" or sample_type == "AUDIO":
        sample.type = projector_pb2.Sample.SampleType.AUDIO
    elif sample_type == "text" or sample_type == "TEXT":
        sample.type = projector_pb2.Sample.SampleType.TEXT
    else:
        raise NotImplementedError
    if x is not None:
        _set_tensor(sample.X, x)
    return
