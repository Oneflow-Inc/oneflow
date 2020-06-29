from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow.customized.utils.plugin_data_pb2 as plugin_data_pb2
import oneflow.customized.utils.summary_pb2 as summary_pb2

PLUGIN_NAME = "hparams"
PLUGIN_DATA_VERSION = 0

EXPERIMENT_TAG = "_hparams_/experiment"
SESSION_START_INFO_TAG = "_hparams_/session_start_info"
SESSION_END_INFO_TAG = "_hparams_/session_end_info"


def create_summary_metadata(hparams_plugin_data_pb):
    """Returns a summary metadata for the HParams plugin.

    Returns a summary_pb2.SummaryMetadata holding a copy of the given
    HParamsPluginData message in its plugin_data.content field.
    Sets the version field of the hparams_plugin_data_pb copy to
    PLUGIN_DATA_VERSION.

    Args:
      hparams_plugin_data_pb: the HParamsPluginData protobuffer to use.
    """
    if not isinstance(hparams_plugin_data_pb, plugin_data_pb2.HParamsPluginData):
        raise TypeError(
            "Needed an instance of plugin_data_pb2.HParamsPluginData."
            " Got: %s" % type(hparams_plugin_data_pb)
        )
    content = plugin_data_pb2.HParamsPluginData()
    content.CopyFrom(hparams_plugin_data_pb)
    content.version = PLUGIN_DATA_VERSION
    return summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )


def parse_session_start_info_plugin_data(content):
    return _parse_plugin_data_as(content, "session_start_info")


def _parse_plugin_data_as(content, data_oneof_field):
    plugin_data = plugin_data_pb2.HParamsPluginData.FromString(content)
    if plugin_data.version != PLUGIN_DATA_VERSION:
        raise Exception(
            "Only supports plugin_data version: %s; found: %s in: %s"
            % (PLUGIN_DATA_VERSION, plugin_data.version, plugin_data)
        )
    if not plugin_data.HasField(data_oneof_field):
        raise Exception(
            "Expected plugin_data.%s to be set. Got: %s"
            % (data_oneof_field, plugin_data)
        )
    return getattr(plugin_data, data_oneof_field)
