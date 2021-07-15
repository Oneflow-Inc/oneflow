"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def PythonDict2CFG(value, msg):
    def extend_dict(values, msg):
        for k, v in values.items():
            if type(v) is dict:
                extend_dict(v, getattr(msg, "mutable_" + k)())
            elif type(v) is list or type(v) is tuple:
                extend_list_or_tuple(v, msg, k)
            else:
                getattr(msg, "set_" + k)(v)

    def extend_list_or_tuple(values, msg, attr):
        if len(values) == 0 or type(values[0]) is dict:
            msg = getattr(msg, "mutable_" + attr)()
            for v in values:
                cmd = msg.Add()
                extend_dict(v, cmd)
        else:
            for v in values:
                getattr(msg, "add_" + attr)(v)

    extend_dict(value, msg)
    return msg


def PythonDict2PbMessage(value, msg):
    def extend_dict(values, msg):
        for k, v in values.items():
            if type(v) is dict:
                extend_dict(v, getattr(msg, k))
            elif type(v) is list or type(v) is tuple:
                extend_list_or_tuple(v, getattr(msg, k))
            else:
                setattr(msg, k, v)
        else:
            msg.SetInParent()

    def extend_list_or_tuple(values, msg):
        if len(values) == 0:
            return
        if type(values[0]) is dict:
            for v in values:
                cmd = msg.add()
                extend_dict(v, cmd)
        else:
            msg.extend(values)

    extend_dict(value, msg)
    return msg


def MergePbMessage(dst, src):
    assert type(dst) is type(src)
    for field in dst.DESCRIPTOR.fields:
        field_name = field.name
        if field.containing_oneof is not None:
            if dst.WhichOneof(field.containing_oneof.name) is not None:
                continue
            src_field_name = src.WhichOneof(field.containing_oneof.name)
            if src_field_name is None:
                continue
            if field_name != src_field_name:
                continue
        else:
            if dst.HasField(field_name):
                continue
            if not src.HasField(field_name):
                continue
        _MergePbMessageField(dst, src, field)


def _MergePbMessageField(dst, src, field):
    if field.message_type is None:
        setattr(dst, field.name, getattr(src, field.name))
    else:
        MergePbMessage(getattr(dst, field.name), getattr(src, field.name))
