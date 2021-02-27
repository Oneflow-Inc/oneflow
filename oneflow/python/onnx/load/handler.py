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
import os
import shutil

from onnx import defs


class BackendHandler:
    """
  All operator handler MUST put decorator @onnx_op to register corresponding op.
  """

    ONNX_OP = None

    DOMAIN = defs.ONNX_DOMAIN
    VERSION = 0
    SINCE_VERSION = 0
    PARTIAL_SUPPORT = False
    PS_DESCRIPTION = ""

    @classmethod
    def check_cls(cls):
        if not cls.ONNX_OP:
            common.logger.warning(
                "{} doesn't have ONNX_OP. "
                "Please use BackendHandler.onnx_op decorator to register ONNX_OP.".format(
                    cls.__name__
                )
            )

    @classmethod
    def handle(cls, node, tensor_dict, **kwargs):
        """ Main method in handler. It will find corresponding versioned handle method,
    whose name format is `version_%d`. So prefix `version_` is reserved in onnx-tensorflow.
    DON'T use it for other purpose.

    :param node: NodeProto for backend.
    :param kwargs: Other args.
    :return: TensorflowNode for backend.
    """
        ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
        if ver_handle:
            return ver_handle(node, tensor_dict, **kwargs)
        raise ValueError(
            'node "{}" of version {} is not supported'.format(
                node.op_type, cls.SINCE_VERSION
            )
        )
        return None

    @classmethod
    def get_versions(cls):
        """ Get all support versions.

    :return: Version list.
    """
        versions = []
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("version_"):
                versions.append(int(k.replace("version_", "")))
        return versions

    @staticmethod
    def onnx_op(op):
        return BackendHandler.property_register("ONNX_OP", op)

    @staticmethod
    def flow_func(func):
        return BackendHandler.property_register("FLOW_FUNC", func)

    @staticmethod
    def domain(d):
        return BackendHandler.property_register("DOMAIN", d)

    @staticmethod
    def partial_support(ps):
        return BackendHandler.property_register("PARTIAL_SUPPORT", ps)

    @staticmethod
    def ps_description(psd):
        return BackendHandler.property_register("PS_DESCRIPTION", psd)

    @staticmethod
    def property_register(name, value):
        def deco(cls):
            setattr(cls, name, value)
            return cls

        return deco

    FLOW_FUNC = None
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
    :param flow_func: Callable OneFlow function. Default is cls.FLOW_FUNC.
    :param inputs: Inputs tensor. Default is got from node.inputs.
    :param attrs: Attributes. Default is node.attrs.
    :param name: Node name.
    :param kwargs: Other args.
    :return: Tensor.
    """
        if flow_func is None:
            flow_func = cls.FLOW_FUNC
        if inputs is None:
            inputs = [tensor_dict.get(inp, None) for inp in node.input_tensor_names]
        if attrs is None:
            attrs = copy.deepcopy(node.attrs)
        if name != "":
            attrs["name"] = name

        return cls._run_flow_func(flow_func, inputs, attrs)

    @classmethod
    def _run_flow_func(cls, flow_func, inputs, attrs):
        """ Run Oneflow function.
    Use only acceptable attributes of function from attrs.

    :param flow_func: Tensorflow function.
    :param inputs: Inputs.
    :param attrs: Attributes.
    :return: Tensor.
    """
        params = list(inspect.signature(flow_func).parameters.keys())

        attrs = cls._process_attrs(attrs)
        attrs = {p: v for p, v in attrs.items() if p in params}
        kwargs = dict(zip(params, inputs))
        ambiguous_arguments = any(
            kwargs.get(p) is not None and v is not None for p, v in attrs.items()
        )
        if ambiguous_arguments:
            raise TypeError("Ambiguous arguments for {}()".format(flow_func.__name__))
        kwargs.update((p, v) for p, v in attrs.items() if v is not None)
        return flow_func(**kwargs)


domain = BackendHandler.domain
onnx_op = BackendHandler.onnx_op
flow_func = BackendHandler.flow_func
partial_support = BackendHandler.partial_support
ps_description = BackendHandler.ps_description
