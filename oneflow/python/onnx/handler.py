# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Opset registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import inspect

from oneflow.python.onnx import constants

# pylint: disable=unused-argument,missing-docstring,invalid-name


class flow_op:
    """Class to implement the decorator to register handlers that map tf to onnx."""

    _OPSETS = collections.OrderedDict()
    _MAPPING = None
    name_set = set()

    def __init__(self, name, onnx_op=None, domain=constants.ONNX_DOMAIN, **kwargs):
        """Called decorator from decorator.

        :param name: The name of the oneflow operator.
        :param domain: The domain the operator belongs to, defaults to onnx.
        :param kwargs: Dictionary that are passed to the handler. A key 'onnx_op' will change the operator name.
        """
        if not isinstance(name, list):
            name = [name]
        self.name = name
        if not isinstance(onnx_op, list):
            onnx_op = [onnx_op] * len(name)
        self.onnx_op = onnx_op
        self.domain = domain
        self.kwargs = kwargs

    def __call__(self, func):
        opset = flow_op._OPSETS.get(self.domain)
        if not opset:
            opset = []
            flow_op._OPSETS[self.domain] = opset
        for k, v in inspect.getmembers(func, inspect.ismethod):
            if k.startswith("version_"):
                version = int(k.replace("version_", ""))
                while version >= len(opset):
                    opset.append({})
                opset_dict = opset[version]
                for i, name in enumerate(self.name):
                    opset_dict[name] = (v, self.onnx_op[i], self.kwargs)
                    flow_op.name_set.add(name)
        print(len(flow_op.name_set))
        return func

    def register_compat_handler(self, func, version):
        """Register old style custom handler.

        :param func: The handler.
        :param version: The domain the operator belongs to, defaults to onnx.
        :param version: The version of the handler.
        """
        opset = flow_op._OPSETS.get(self.domain)
        if not opset:
            opset = []
            flow_op._OPSETS[self.domain] = opset
            while version >= len(opset):
                opset.append({})
            opset_dict = opset[version]
            opset_dict[self.name[0]] = (func, self.onnx_op[0], self.kwargs)

    @staticmethod
    def get_opsets():
        return flow_op._OPSETS

    @staticmethod
    def create_mapping(max_onnx_opset_version, extra_opsets):
        """Create the final mapping dictionary by stacking domains and opset versions.

        :param max_onnx_opset_version: The highest onnx opset the resulting graph may use.
        :param extra_opsets: Extra opsets the resulting graph may use.
        """
        mapping = {constants.ONNX_DOMAIN: max_onnx_opset_version}
        if extra_opsets:
            for extra_opset in extra_opsets:
                mapping[extra_opset.domain] = extra_opset.version
        ops_mapping = {}
        for domain, opsets in flow_op.get_opsets().items():
            for target_opset, op_map in enumerate(opsets):
                m = mapping.get(domain)
                if m:
                    if target_opset <= m and op_map:
                        ops_mapping.update(op_map)

        flow_op._MAPPING = ops_mapping
        return ops_mapping

    @staticmethod
    def find_effective_op(name):
        """Find the effective version of an op create_mapping.
           This is used if we need to compose ops from other ops where we'd need to find the
           op that is doing to be used in the final graph, for example there is a custom op
           that overrides a onnx op ...

        :param name: The operator name.
        """
        map_info = flow_op._MAPPING.get(name)
        if map_info is None:
            return None
        return map_info
