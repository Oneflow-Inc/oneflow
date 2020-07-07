from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import hashlib
import json
import time

import oneflow.customized.utils.plugin_data_pb2 as plugin_data_pb2
import oneflow.customized.utils.summary_pb2 as summary_pb2
import oneflow.customized.utils.event_pb2 as event_pb2
import oneflow.customized.utils.tensor_pb2 as tensor_pb2
import oneflow.customized.utils.projector_pb2 as projector_pb2
from oneflow.python.oneflow_export import oneflow_export


import oneflow as flow


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
def text(text, tag=None):
    if isinstance(text, (tuple, list)) and len(text) > 0:
        if not isinstance(tag, str) or tag is None:
            tag = "text"
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
            tag=tag,
            metadata=summary_pb2.SummaryMetadata(
                plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="text")
            ),
            tensor=tensor,
        )
        return summary


def _get_tensor(values, dtype=None, shape=None):
    array = np.empty(shape, dtype=np.float)
    tensor_shape = tensor_pb2.TensorShapeProto()
    dim = tensor_shape.dim.add()
    dim.size = 0

    tensor_proto = tensor_pb2.TensorProto(
        dtype=tensor_pb2.DT_FLOAT, tensor_shape=tensor_shape,
    )
    proto_values = array.ravel()
    tensor_proto.float_val.extend([np.asscalar(x) for x in proto_values])
    return tensor_proto


@oneflow_export("hparams")
def hparams(hparams):
    hparams, metrics = _get_hparams_dict(hparams)
    jparams = json.dumps(hparams, sort_keys=True, separators=(",", ":"))
    group_name = hashlib.sha256(jparams.encode("utf-8")).hexdigest()

    session_start_info = plugin_data_pb2.SessionStartInfo(
        group_name=group_name, start_time_secs=time.time(),
    )
    for key in sorted(hparams):
        value = hparams[key]
        if isinstance(value, str):
            session_start_info.hparams[key].string_value = value
        elif isinstance(value, (float, int)):
            session_start_info.hparams[key].number_value = value
        elif isinstance(value, bool):
            session_start_info.hparams[key].bool_value = value
        else:
            raise TypeError("the type of value: %r is not supported!" % value)
    for key in metrics:
        value = metrics[key]
        if isinstance(value, (float, int)):
            session_start_info.metrics[key].number_value = value
        else:
            raise TypeError("the type of value: %r is not supported!" % value)

    summary = summary_pb2.Summary()
    summary_metadata = _get_metadata(
        plugin_data_pb2.HParamsPluginData(session_start_info=session_start_info)
    )
    summary.value.add(
        tag="_hparams_/session_start_info",
        metadata=summary_metadata,
        tensor=_get_tensor([], tensor_pb2.DT_FLOAT, (0,)),
    )
    return summary


def _get_metadata(hparams_plugin_data):
    plugin_data = plugin_data_pb2.HParamsPluginData()
    plugin_data.CopyFrom(hparams_plugin_data)
    plugin_data.version = 0
    return summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name="hparams", content=plugin_data.SerializeToString()
        )
    )


def _get_hparams_dict(hparams):
    hparams_dict = {}
    metrics_dict = {}
    for (key, value) in dict.items(hparams):
        if key in hparams_dict or key in metrics_dict:
            raise ValueError("the key is already exist %r" % (key,))
        if isinstance(key, HParam):
            key = key.name
        if isinstance(key, Metric):
            metrics_dict[key.name] = _get_value(value)
            continue      
        hparams_dict[key] = _get_value(value)
    return hparams_dict, metrics_dict


def _get_value(value):
    if isinstance(value, np.generic):
        return value.item()
    else:
        return value

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
        if dtype not in (int, float, bool, str):
            raise ValueError(
                "Value type must in (int, float, bool, str), %r is not supported!"
                % (dtype,)
            )
        self.dtype_ = dtype
        for value in self.values_:
            if not isinstance(value, self.dtype_):
                raise TypeError(
                    "The type of value is not supported! value: %r type: %s"
                    % (value, self.dtype_.__name__)
                )
        self.values_.sort()

    @property
    def dtype(self):
        return self.dtype_

    @property
    def values(self):
        return list(self.values_)


@oneflow_export("Metric")
class Metric(object):
    def __init__(self, name, dtype=None):
        self.name_ = name
        if dtype is None:
            dtype = float
        if dtype not in (int, float):
            raise ValueError(
                "Value type must in (int, float), %r is not supported!" % (dtype,)
            )
        self.dtype_ = dtype

    @property
    def name(self):
        return self.name_

    @property
    def dtype(self):
        return self.dtype_
