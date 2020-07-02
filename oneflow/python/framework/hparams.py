from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import abc
import hashlib
import random
import json
import time

import oneflow.customized.utils.plugin_data_pb2 as plugin_data_pb2
import oneflow.customized.utils.summary_pb2 as summary_pb2
import oneflow.python.test.customized.metadata as metadata
import oneflow.customized.utils.event_pb2 as event_pb2
import oneflow.customized.utils.tensor_pb2 as tensor_pb2
import oneflow.customized.utils.projector_pb2 as projector_pb2
from oneflow.python.oneflow_export import oneflow_export


import oneflow as flow


@oneflow_export("exception_projector")
def exception_projector():
    value = np.random.rand(100,).astype(np.float32)
    summary_projector = projector_pb2.SummaryProjector()
    summary_projector.metadata.type = projector_pb2.MetaData.ProjectorType.EXCEPTION
    projector = summary_projector.projector.add()
    set_projector(projector, "tag1", 10, value)
    filename = "/home/zjhushengjian/oneflow/projector.gradient.1593592184.v2"
    print(summary_projector)
    with open(filename, "wb") as f:
        f.write(summary_projector.SerializeToString())
        f.flush()


def set_tensor(tensor: projector_pb2.Tensor, value):
    for d in value.shape:
        td = tensor.shape.dim.add()
        td.size = d
    tensor.dtype = str(value.dtype)
    tensor.content = value.tobytes()
    return


def set_projector(pro, tag, step, value, label=None):
    pro.tag = str(tag)
    pro.step = step
    pro.WALL_TIME = time.time()
    set_tensor(pro.value, value)
    return


def as_bytes(bytes_or_text, encoding="utf-8"):
    if isinstance(bytes_or_text, bytearray):
        return bytes(bytes_or_text)
    elif isinstance(bytes_or_text, six.text_type):
        return bytes_or_text.encode(encoding)
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text
    else:
        raise TypeError("Expected binary or unicode string, got %r" % (bytes_or_text,))


# write text
@oneflow_export("text")
def text(text):
    if isinstance(text, (tuple, list)) and len(text) > 0:
        text_size = len(text)
        tensor_shape = tensor_pb2.TensorShapeProto()
        dim = tensor_shape.dim.add()
        dim.size = text_size

        tensor = tensor_pb2.TensorProto(
            dtype=tensor_pb2.DT_STRING, tensor_shape=tensor_shape,
        )
        for idx in range(text_size):
            tensor.string_val.append(as_bytes(text[idx]))  # str.encode(text[idx]))
        summary = summary_pb2.Summary()
        value = summary.value.add(
            tag="text",
            metadata=summary_pb2.SummaryMetadata(
                plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="text")
            ),
            tensor=tensor,
        )
        return summary


def AddFloatsToProto(proto, values):
    proto.float_val.extend([np.asscalar(x) for x in values])


def make_tensor_proto(values, dtype=None, shape=None):
    nparray = np.empty(shape, dtype=np.float)
    tshape = tensor_pb2.TensorShapeProto()
    dim = tshape.dim.add()
    dim.size = 0

    tensor_proto = tensor_pb2.TensorProto(
        dtype=tensor_pb2.DT_FLOAT, tensor_shape=tshape,
    )
    proto_values = nparray.ravel()
    AddFloatsToProto(tensor_proto, proto_values)
    return tensor_proto


NULL_TENSOR = make_tensor_proto([], tensor_pb2.DT_FLOAT, (0,))


@oneflow_export("hparams_pb")
def hparams_pb(hparams, trial_id=None, start_time_secs=None):
    if start_time_secs is None:
        start_time_secs = time.time()
    hparams = _normalize_hparams(hparams)
    group_name = _derive_session_group_name(trial_id, hparams)

    session_start_info = plugin_data_pb2.SessionStartInfo(
        group_name=group_name, start_time_secs=start_time_secs,
    )
    for hp_name in sorted(hparams):
        hp_value = hparams[hp_name]
        if isinstance(hp_value, bool):
            session_start_info.hparams[hp_name].bool_value = hp_value
        elif isinstance(hp_value, (float, int)):
            session_start_info.hparams[hp_name].number_value = hp_value
        elif isinstance(hp_value, six.string_types):
            session_start_info.hparams[hp_name].string_value = hp_value
        else:
            raise TypeError(
                "hparams[%r] = %r, of unsupported type %r"
                % (hp_name, hp_value, type(hp_value))
            )

    return _summary_pb(
        metadata.SESSION_START_INFO_TAG,
        plugin_data_pb2.HParamsPluginData(session_start_info=session_start_info),
        NULL_TENSOR,
    )


def _normalize_hparams(hparams):
    result = {}
    for (k, v) in six.iteritems(hparams):
        if isinstance(k, HParam):
            k = k.name
        if k in result:
            raise ValueError("multiple values specified for hparam %r" % (k,))
        result[k] = _normalize_numpy_value(v)
    return result


def _normalize_numpy_value(value):
    if isinstance(value, np.generic):
        return value.item()
    else:
        return value


def _derive_session_group_name(trial_id, hparams):
    if trial_id is not None:
        if not isinstance(trial_id, six.string_types):
            raise TypeError("`trial_id` should be a `str`, but got: %r" % (trial_id,))
        return trial_id
    jparams = json.dumps(hparams, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(jparams.encode("utf-8")).hexdigest()


def _summary_pb(tag, hparams_plugin_data, tensor):
    summary = summary_pb2.Summary()
    summary_metadata = metadata.create_summary_metadata(hparams_plugin_data)
    value = summary.value.add(tag=tag, metadata=summary_metadata, tensor=tensor)
    return summary


@oneflow_export("Hparam")
class HParam(object):
    def __init__(self, name, dtype=None):
        self.name_ = name
        self.dtype_ = dtype
        if not isinstance(self.dtype_, (IntegerRange, RealRange, ValueSet, type(None))):
            raise ValueError(
                "Hparam dtype must be: (IntegerRange, RealRange, ValueSet) : %r"
                % (self.dtype_,)
            )

    def __repr__(self):
        hparam_info = [
            ("name", self.name_),
            ("dtype", self.dtype_),
        ]
        for (key, value) in hparam_info:
            hparam_str = ", ".join("%s=%r" % (key, value))
        return "HParam(%s)" % hparam_str

    @property
    def name(self):
        return self.name_

    @property
    def dtype(self):
        return self.dtype_


@oneflow_export("IntegerRange")
class IntegerRange(object):
    def __init__(self, min_value, max_value):
        if not isinstance(max_value, int):
            raise TypeError("max_value is not an integer value: %r" % (max_value,))
        if not isinstance(min_value, int):
            raise TypeError("min_value is not an integer value: %r" % (min_value,))
        if min_value > max_value:
            raise ValueError(
                "max_value must bigger than min_value: %r > %r" % (min_value, max_value)
            )
        self.min_value_ = min_value
        self.max_value_ = max_value

    def __repr__(self):
        return "IntegerRange(%r, %r)" % (self.min_value_, self.max_value_)

    @property
    def min_value(self):
        return self.min_value_

    @property
    def max_value(self):
        return self.max_value_


@oneflow_export("RealRange")
class RealRange(object):
    def __init__(self, min_value, max_value):
        if not isinstance(max_value, float):
            raise TypeError("max_value is not an float value: %r" % (max_value,))
        if not isinstance(min_value, float):
            raise TypeError("min_value is not an float value: %r" % (min_value,))
        if min_value > max_value:
            raise ValueError(
                "max_value must bigger than min_value: %r > %r" % (min_value, max_value)
            )
        self.min_value_ = min_value
        self.max_value_ = max_value

    def __repr__(self):
        return "RealRange(%r, %r)" % (self.min_value_, self.max_value_)

    @property
    def min_value(self):
        return self.min_value_

    @property
    def max_value(self):
        return self.max_value_


@oneflow_export("ValueSet")
class ValueSet(object):
    def __init__(self, values, dtype=None):
        self.values_ = list(values)
        if dtype is None:
            if self.values_:
                dtype = type(self.values_[0])
            else:
                raise ValueError("Empty domain with no dtype specified")
        if dtype not in (int, float, bool, str):
            raise ValueError("Unknown dtype: %r" % (dtype,))
        self.dtype_ = dtype
        for value in self.values_:
            if not isinstance(value, self.dtype_):
                raise TypeError(
                    "dtype mismatch: not isinstance(%r, %s)"
                    % (value, self.dtype_.__name__)
                )
        self.values_.sort()

    def __repr__(self):
        return "ValueSet(%r)" % (self.values_,)

    @property
    def dtype(self):
        return self.dtype_

    @property
    def values(self):
        return list(self.values_)


@oneflow_export("Metric")
class Metric(object):
    """A metric in an experiment.

    A metric is a real-valued function of a model. Each metric is
    associated with a TensorBoard scalar summary, which logs the
    metric's value as the model trains.
    """

    TRAINING = 1
    VALIDATION = 2

    def __init__(
        self, tag, group=None, display_name=None, description=None, dataset_type=None,
    ):
        """

        Args:
          tag: The tag name of the scalar summary that corresponds to this
            metric (as a `str`).
          group: An optional string listing the subdirectory under the
            session's log directory containing summaries for this metric.
            For instance, if summaries for training runs are written to
            events files in `ROOT_LOGDIR/SESSION_ID/train`, then `group`
            should be `"train"`. Defaults to the empty string: i.e.,
            summaries are expected to be written to the session logdir.
          display_name: An optional human-readable display name.
          description: An optional Markdown string with a human-readable
            description of this metric, to appear in TensorBoard.
          dataset_type: Either `Metric.TRAINING` or `Metric.VALIDATION`, or
            `None`.
        """
        self._tag = tag
        self._group = group
        self._display_name = display_name
        self._description = description
        self._dataset_type = dataset_type
        if self._dataset_type not in (None, Metric.TRAINING, Metric.VALIDATION):
            raise ValueError("invalid dataset type: %r" % (self._dataset_type,))

    def as_proto(self):
        return api_pb2.MetricInfo(
            name=api_pb2.MetricName(group=self._group, tag=self._tag,),
            display_name=self._display_name,
            description=self._description,
            dataset_type=self._dataset_type,
        )
