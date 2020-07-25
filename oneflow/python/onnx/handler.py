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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Opset registry

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import inspect

from oneflow.python.onnx import constants

# pylint: disable=unused-argument,missing-docstring,invalid-name


class flow_op:
    """Class to implement the decorator to register handlers that map oneflow to onnx."""

    _OPSETS = collections.OrderedDict()
    _MAPPING = None
    _OP_TYPE_2_IBN = {}
    _OP_TYPE_2_OBN = {}
    name_set = set()

    def __init__(
        self,
        name,
        onnx_op=None,
        domain=constants.ONNX_DOMAIN,
        flow_ibns=None,
        flow_obns=None,
        **kwargs
    ):
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
        self.flow_ibns = flow_ibns
        self.flow_obns = flow_obns

    def __call__(self, func):
        opset = flow_op._OPSETS.get(self.domain)
        if not opset:
            opset = []
            flow_op._OPSETS[self.domain] = opset
        for k, v in inspect.getmembers(func, inspect.ismethod):
            if k.startswith("Version_"):
                version = int(k.replace("Version_", ""))
                while version >= len(opset):
                    opset.append({})
                opset_dict = opset[version]
                for i, name in enumerate(self.name):
                    opset_dict[name] = (v, self.onnx_op[i], self.kwargs)
                    flow_op.name_set.add(name)
                    if self.flow_ibns is not None:
                        flow_op._OP_TYPE_2_IBN[name] = self.flow_ibns
                    if self.flow_obns is not None:
                        flow_op._OP_TYPE_2_OBN[name] = self.flow_obns
        return func

    @staticmethod
    def ibn4op_type(op_type):
        return flow_op._OP_TYPE_2_IBN.get(op_type, None)

    @staticmethod
    def obn4op_type(op_type):
        return flow_op._OP_TYPE_2_OBN.get(op_type, None)

    @staticmethod
    def get_opsets():
        return flow_op._OPSETS

    @staticmethod
    def CreateMapping(max_onnx_opset_version, extra_opsets):
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
