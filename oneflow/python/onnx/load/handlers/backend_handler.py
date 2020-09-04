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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import inspect

import tensorflow as tf

from oneflow.python.onnx.load.common import IS_PYTHON3
from oneflow.python.onnx.load.common import get_data_format
from oneflow.python.onnx.load.common import get_perm_from_formats
from oneflow.python.onnx.load.common import supports_device
from oneflow.python.onnx.handler import Handler
import os
import shutil


class BackendHandler(Handler):
    """ This class is base backend handler class.
  All backend operator handler class MUST inherit this class.
  In backend, operator handler class's name should be pascal case of file name
  which should be snake case.
  Use ONNX operator name as class name.
  """

    TF_FUNC = None
    WEIGHT_SAVE_DIR = None

    @classmethod
    def copy_variable_file(cls, src_var_name, dst_var_name):
        dst_dir_name = os.path.join(cls.WEIGHT_SAVE_DIR, dst_var_name)
        if not os.path.exists(dst_dir_name):
            os.makedirs(dst_dir_name)
        shutil.copyfile(
            os.path.join(cls.WEIGHT_SAVE_DIR, src_var_name, "out"),
            os.path.join(dst_dir_name, "out"),
        )

    @classmethod
    def get_attrs_processor_param(cls):
        """ Get param for attrs processor.

    :return: Dict.
    """
        return {}

    @classmethod
    def _process_attrs(cls, attrs):
        """ Private method for processing attrs.
    Param for this processor got from `get_attrs_processor_param`.
    Param is dict contains two key: `default` and `raname`.
    First add default value to attrs if key does not exist.
    Second rename key to new key.

    For example:
      attrs = {"keep_dims": True}
      param = {"default": {"axis": 1},
               "rename": {"keep_dims": "keepdims"}}

      processed_attrs = {"axis": "1", "keepdims": True}

    :param attrs: Process target attrs.
    :return: Processed attrs.
    """
        param = {"rename": {}, "default": {}}
        param.update(cls.get_attrs_processor_param())

        for k, v in param["default"].items():
            attrs.setdefault(k, v)

        for k, new_k in param["rename"].items():
            if k in attrs:
                attrs[new_k] = attrs.pop(k)

        return attrs

    @classmethod
    def run_onnx_node(
        cls,
        node,
        tensor_dict,
        flow_func=None,
        inputs=None,
        attrs=None,
        name="",
        **kwargs
    ):
        """ Helper method to make tensor.

    :param node: OnnxNode object.
    :param tf_func: Callable Tf function. Default is cls.TF_FUNC.
    :param inputs: Inputs tensor. Default is got from node.inputs.
    :param attrs: Attributes. Default is node.attrs.
    :param name: Node name.
    :param kwargs: Other args.
    :return: Tensor.
    """
        if flow_func is None:
            flow_func = cls.TF_FUNC
        if inputs is None:
            inputs = [tensor_dict.get(inp, None) for inp in node.input_tensor_names]
        if attrs is None:
            attrs = copy.deepcopy(node.attrs)
        name = name or node.name
        if name != "":
            attrs["name"] = name

        return cls._run_tf_func(flow_func, inputs, attrs)

    @classmethod
    def _run_tf_func(cls, tf_func, inputs, attrs):
        """ Run Tensorflow function.
    Use only acceptable attributes of function from attrs.

    :param tf_func: Tensorflow function.
    :param inputs: Inputs.
    :param attrs: Attributes.
    :return: Tensor.
    """
        if IS_PYTHON3:
            params = list(inspect.signature(tf_func).parameters.keys())
        else:
            # use closure to get args for function using decorator
            if tf_func.__closure__ is not None:
                while "__wrapped__" in tf_func.func_dict:
                    tf_func = tf_func.func_dict["__wrapped__"]
                params = inspect.getargspec(tf_func).args
            else:
                params = inspect.getargspec(tf_func).args

        attrs = cls._process_attrs(attrs)
        attrs = {p: v for p, v in attrs.items() if p in params}
        kwargs = dict(zip(params, inputs))
        ambiguous_arguments = any(
            kwargs.get(p) is not None and v is not None for p, v in attrs.items()
        )
        if ambiguous_arguments:
            raise TypeError("Ambiguous arguments for {}()".format(tf_func.__name__))
        kwargs.update((p, v) for p, v in attrs.items() if v is not None)
        return tf_func(**kwargs)
