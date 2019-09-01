from __future__ import absolute_import

from google.protobuf import text_format

class JobBuildAndInferError(Exception):
    def __init__(self, error_proto):
        assert error_proto.HasField("error_type")
        self.error_proto_ = error_proto

    def __str__(self):
        return text_format.MessageToString(self.error_proto_)
