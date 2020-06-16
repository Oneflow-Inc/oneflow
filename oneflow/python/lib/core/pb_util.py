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
