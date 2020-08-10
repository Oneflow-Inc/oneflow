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

import inspect

from onnx import defs

from oneflow.python.onnx.load.common import exception
from oneflow.python.onnx.load.common import IS_PYTHON3


class Handler(object):
    """ This class is base handler class.
  Base backend and frontend base handler class inherit this class.

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
                "Please use Handler.onnx_op decorator to register ONNX_OP.".format(
                    cls.__name__
                )
            )

    @classmethod
    def args_check(cls, node, **kwargs):
        """ Check args. e.g. if shape info is in graph.
    Raise exception if failed.

    :param node: NodeProto for backend.
    :param kwargs: Other args.
    """
        pass

    @classmethod
    def handle(cls, node, **kwargs):
        """ Main method in handler. It will find corresponding versioned handle method,
    whose name format is `version_%d`. So prefix `version_` is reserved in onnx-tensorflow.
    DON'T use it for other purpose.

    :param node: NodeProto for backend.
    :param kwargs: Other args.
    :return: TensorflowNode for backend.
    """
        ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
        if ver_handle:
            cls.args_check(node, **kwargs)
            return ver_handle(node, **kwargs)
        exception.OP_UNIMPLEMENTED_EXCEPT(node.op_type, cls.SINCE_VERSION)
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
        return Handler.property_register("ONNX_OP", op)

    @staticmethod
    def tf_func(func):
        return Handler.property_register("TF_FUNC", func)

    @staticmethod
    def domain(d):
        return Handler.property_register("DOMAIN", d)

    @staticmethod
    def partial_support(ps):
        return Handler.property_register("PARTIAL_SUPPORT", ps)

    @staticmethod
    def ps_description(psd):
        return Handler.property_register("PS_DESCRIPTION", psd)

    @staticmethod
    def property_register(name, value):
        def deco(cls):
            if inspect.isfunction(value) and not IS_PYTHON3:
                setattr(cls, name, staticmethod(value))
            else:
                setattr(cls, name, value)
            return cls

        return deco


domain = Handler.domain
onnx_op = Handler.onnx_op
tf_func = Handler.tf_func
partial_support = Handler.partial_support
ps_description = Handler.ps_description
property_register = Handler.property_register
