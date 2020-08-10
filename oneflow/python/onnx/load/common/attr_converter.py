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
from oneflow.python.onnx.load.common import IS_PYTHON3


def convert_tf(attr):
    return __convert_tf_attr_value(attr)


def convert_onnx(attr):
    return __convert_onnx_attribute_proto(attr)


def __convert_tf_attr_value(attr):
    """ convert Tensorflow AttrValue object to Python object
  """
    if attr.HasField("list"):
        return __convert_tf_list_value(attr.list)
    if attr.HasField("s"):
        return attr.s
    elif attr.HasField("i"):
        return attr.i
    elif attr.HasField("f"):
        return attr.f
    elif attr.HasField("b"):
        return attr.b
    elif attr.HasField("type"):
        return attr.type
    elif attr.HasField("shape"):
        return attr.type
    elif attr.HasField("tensor"):
        return attr.tensor
    else:
        raise ValueError("Unsupported Tensorflow attribute: {}".format(attr))


def __convert_tf_list_value(list_value):
    """ convert Tensorflow ListValue object to Python object
  """
    if list_value.s:
        return list_value.s
    elif list_value.i:
        return list_value.i
    elif list_value.f:
        return list_value.f
    elif list_value.b:
        return list_value.b
    elif list_value.tensor:
        return list_value.tensor
    elif list_value.type:
        return list_value.type
    elif list_value.shape:
        return list_value.shape
    elif list_value.func:
        return list_value.func
    else:
        raise ValueError("Unsupported Tensorflow attribute: {}".format(list_value))


def __convert_onnx_attribute_proto(attr_proto):
    """
  Convert an ONNX AttributeProto into an appropriate Python object
  for the type.
  NB: Tensor attribute gets returned as the straight proto.
  """
    if attr_proto.HasField("f"):
        return attr_proto.f
    elif attr_proto.HasField("i"):
        return attr_proto.i
    elif attr_proto.HasField("s"):
        return str(attr_proto.s, "utf-8") if IS_PYTHON3 else attr_proto.s
    elif attr_proto.HasField("t"):
        return attr_proto.t  # this is a proto!
    elif attr_proto.HasField("g"):
        return attr_proto.g
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        if IS_PYTHON3:
            str_list = list(map(lambda x: str(x, "utf-8"), str_list))
        return str_list
    elif attr_proto.HasField("sparse_tensor"):
        return attr_proto.sparse_tensor
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))
