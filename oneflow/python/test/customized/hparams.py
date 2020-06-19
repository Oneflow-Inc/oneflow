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
from oneflow.python.oneflow_export import oneflow_export


import oneflow as flow

@oneflow_export("hparams")
def hparams(hparams, trial_id=None, start_time_secs=None):
    pb = hparams_pb(
        hparams=hparams, trial_id=trial_id, start_time_secs=start_time_secs,
    )
    return _write_summary("hparams", pb)

def _write_summary(name, pb):
    """Write a summary, returning the writing op.

    Args:
      name: As passed to `summary_scope`.
      pb: A `summary_pb2.Summary` message.

    Returns:
      A tensor whose value is `True` on success, or `False` if no summary
      was written because no default summary writer was available.
    """
    raw_pb = pb.SerializeToString()

    event = event_pb2.Event(summary=pb)
    event.wall_time = 22222
    event.step = 0
    #event.summary = summary_pb2.Summary(pb)

    event_pb = event.SerializeToString()
    
    filename = "/home/zjhushengjian/oneflow/events.out.tfevents.2222.oneflow-15.v2"
    with open(filename, 'wb') as f:
        f.write(event_pb)


    # summary_scope = (
    #     getattr(tf.summary.experimental, "summary_scope", None)
    #     or tf.summary.summary_scope
    # )
    # with summary_scope(name):
    #     return tf.summary.experimental.write_raw_pb(raw_pb, step=0)
    
    # flow.clear_default_session()
    # func_config = flow.FunctionConfig()
    # func_config.default_data_type(flow.double)
    # @flow.function(func_config)
    # def HparamJob(x=flow.FixedTensorDef(len(raw_pb), ), dtype=flow.kInt8):
    #     flow.summary.hparam(x, step=0)
    # HparamJob(np.array(list(raw_pb.encode("ascii")), dtype=np.int8))

    #return flow.summary.hparam(, step=0)

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
        plugin_data_pb2.HParamsPluginData(
            session_start_info=session_start_info
        ),
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
            raise TypeError(
                "`trial_id` should be a `str`, but got: %r" % (trial_id,)
            )
        return trial_id
    jparams = json.dumps(hparams, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(jparams.encode("utf-8")).hexdigest()



def _summary_pb(tag, hparams_plugin_data):
    summary = summary_pb2.Summary()
    summary_metadata = metadata.create_summary_metadata(hparams_plugin_data)
    value = summary.value.add(
        tag=tag, metadata=summary_metadata, simple_value=1.0
    )
    return summary

@oneflow_export("Hparam")
class HParam(object):
    def __init__(self, name, domain=None, display_name=None, description=None):
        self._name = name
        self._domain = domain
        self._display_name = display_name
        self._description = description
        if not isinstance(self._domain, (Domain, type(None))):
            raise ValueError("not a domain: %r" % (self._domain,))
    def __str__(self):
        return "<HParam %r: %s>" % (self._name, self._domain)

    def __repr__(self):
        fields = [
            ("name", self._name),
            ("domain", self._domain),
            ("display_name", self._display_name),
            ("description", self._description),
        ]
        fields_string = ", ".join("%s=%r" % (k, v) for (k, v) in fields)
        return "HParam(%s)" % fields_string

    @property
    def name(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    @property
    def display_name(self):
        return self._display_name

    @property
    def description(self):
        return self._description


@six.add_metaclass(abc.ABCMeta)
class Domain(object):
    @abc.abstractproperty
    def dtype(self):
        """Data type of this domain: `float`, `int`, `str`, or `bool`."""
        pass

    @abc.abstractmethod
    def sample_uniform(self, rng=random):
        """Sample a value from this domain uniformly at random.

        Args:
          rng: A `random.Random` interface; defaults to the `random` module
            itself.

        Raises:
          IndexError: If the domain is empty.
        """
        pass

    @abc.abstractmethod
    def update_hparam_info(self, hparam_info):
        """Update an `HParamInfo` proto to include this domain.

        This should update the `type` field on the proto and exactly one of
        the `domain` variants on the proto.

        Args:
          hparam_info: An `api_pb2.HParamInfo` proto to modify.
        """
        pass

@oneflow_export("IntInterval")
class IntInterval(Domain):
    """A domain that takes on all integer values in a closed interval."""

    def __init__(self, min_value=None, max_value=None):
        """Create an `IntInterval`.

        Args:
          min_value: The lower bound (inclusive) of the interval.
          max_value: The upper bound (inclusive) of the interval.

        Raises:
          TypeError: If `min_value` or `max_value` is not an `int`.
          ValueError: If `min_value > max_value`.
        """
        if not isinstance(min_value, int):
            raise TypeError("min_value must be an int: %r" % (min_value,))
        if not isinstance(max_value, int):
            raise TypeError("max_value must be an int: %r" % (max_value,))
        if min_value > max_value:
            raise ValueError("%r > %r" % (min_value, max_value))
        self._min_value = min_value
        self._max_value = max_value

    def __str__(self):
        return "[%s, %s]" % (self._min_value, self._max_value)

    def __repr__(self):
        return "IntInterval(%r, %r)" % (self._min_value, self._max_value)

    @property
    def dtype(self):
        return int

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    def sample_uniform(self, rng=random):
        return rng.randint(self._min_value, self._max_value)

    def update_hparam_info(self, hparam_info):
        hparam_info.type = (
            api_pb2.DATA_TYPE_FLOAT64
        )  # TODO(#1998): Add int dtype.
        hparam_info.domain_interval.min_value = self._min_value
        hparam_info.domain_interval.max_value = self._max_value

@oneflow_export("RealInterval")
class RealInterval(Domain):
    """A domain that takes on all real values in a closed interval."""

    def __init__(self, min_value=None, max_value=None):
        """Create a `RealInterval`.

        Args:
          min_value: The lower bound (inclusive) of the interval.
          max_value: The upper bound (inclusive) of the interval.

        Raises:
          TypeError: If `min_value` or `max_value` is not an `float`.
          ValueError: If `min_value > max_value`.
        """
        if not isinstance(min_value, float):
            raise TypeError("min_value must be a float: %r" % (min_value,))
        if not isinstance(max_value, float):
            raise TypeError("max_value must be a float: %r" % (max_value,))
        if min_value > max_value:
            raise ValueError("%r > %r" % (min_value, max_value))
        self._min_value = min_value
        self._max_value = max_value

    def __str__(self):
        return "[%s, %s]" % (self._min_value, self._max_value)

    def __repr__(self):
        return "RealInterval(%r, %r)" % (self._min_value, self._max_value)

    @property
    def dtype(self):
        return float

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    def sample_uniform(self, rng=random):
        return rng.uniform(self._min_value, self._max_value)

    def update_hparam_info(self, hparam_info):
        hparam_info.type = api_pb2.DATA_TYPE_FLOAT64
        hparam_info.domain_interval.min_value = self._min_value
        hparam_info.domain_interval.max_value = self._max_value

@oneflow_export("Discrete")
class Discrete(Domain):
    """A domain that takes on a fixed set of values.

    These values may be of any (single) domain type.
    """

    def __init__(self, values, dtype=None):
        """Construct a discrete domain.

        Args:
          values: A iterable of the values in this domain.
          dtype: The Python data type of values in this domain: one of
            `int`, `float`, `bool`, or `str`. If `values` is non-empty,
            `dtype` may be `None`, in which case it will be inferred as the
            type of the first element of `values`.

        Raises:
          ValueError: If `values` is empty but no `dtype` is specified.
          ValueError: If `dtype` or its inferred value is not `int`,
            `float`, `bool`, or `str`.
          TypeError: If an element of `values` is not an instance of
            `dtype`.
        """
        self._values = list(values)
        if dtype is None:
            if self._values:
                dtype = type(self._values[0])
            else:
                raise ValueError("Empty domain with no dtype specified")
        if dtype not in (int, float, bool, str):
            raise ValueError("Unknown dtype: %r" % (dtype,))
        self._dtype = dtype
        for value in self._values:
            if not isinstance(value, self._dtype):
                raise TypeError(
                    "dtype mismatch: not isinstance(%r, %s)"
                    % (value, self._dtype.__name__)
                )
        self._values.sort()

    def __str__(self):
        return "{%s}" % (", ".join(repr(x) for x in self._values))

    def __repr__(self):
        return "Discrete(%r)" % (self._values,)

    @property
    def dtype(self):
        return self._dtype

    @property
    def values(self):
        return list(self._values)

    def sample_uniform(self, rng=random):
        return rng.choice(self._values)

    def update_hparam_info(self, hparam_info):
        hparam_info.type = {
            int: api_pb2.DATA_TYPE_FLOAT64,  # TODO(#1998): Add int dtype.
            float: api_pb2.DATA_TYPE_FLOAT64,
            bool: api_pb2.DATA_TYPE_BOOL,
            str: api_pb2.DATA_TYPE_STRING,
        }[self._dtype]
        hparam_info.ClearField("domain_discrete")
        hparam_info.domain_discrete.extend(self._values)

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
        self,
        tag,
        group=None,
        display_name=None,
        description=None,
        dataset_type=None,
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
