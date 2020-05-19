from __future__ import absolute_import

from google.protobuf import text_format

class JobBuildAndInferError(Exception):
    def __init__(self, error_proto):
        assert error_proto.HasField("error_type")
        self.error_proto_ = error_proto
        self.msg_ = self.error_proto_.msg
        self.error_proto_.ClearField('msg')

    def __str__(self):
        ret = "\nmsg: %s"%self.msg_
        ret += "\n%s"%text_format.MessageToString(self.error_proto_)
        return ret
