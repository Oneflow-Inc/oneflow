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

# oneflow.python.onnx.graph_helper - class to help building graph, such as helping to make complex node

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import logging
from oneflow.python.framework import id_util
from oneflow.python.onnx import util


# pylint: disable=missing-docstring


logger = logging.getLogger(__name__)


class GraphBuilder(object):
    """help to build graph"""

    def __init__(self, graph):
        self._g = graph

    @property
    def graph(self):
        return self._g

    def MakeSlice(self, kwargs, name=None, shapes=None, dtypes=None):
        """
        slice changes its schema at opset 10: it treats some attributes as dynamic input
        so this function has to process inputs according to graph's opset version
        to get "inputs" and "attr" to feed "MakeNode"
        kwargs: key could be ["data", "starts", "ends", "axes", "steps", "outputs"].
        """
        outputs = kwargs.pop("outputs", None)

        if self.graph.opset < 10:
            # "data" is string
            # "starts", "ends" and "axes" are attributes, and "axes" is optional.
            inputs = [kwargs.pop("data")]
            starts = self.ConvertToAttribute(kwargs.pop("starts"))
            ends = self.ConvertToAttribute(kwargs.pop("ends"))
            axes = self.ConvertToAttribute(kwargs.pop("axes", None), is_optional=True)
            attr = {"starts": starts, "ends": ends, "axes": axes}
        else:
            # slice-10 has 3 required inputs "data", "starts", "ends"l
            # and 2 optional inputs "axes", "steps"
            # input sequence should be "data", "starts", "ends", "axes", "steps"
            attr = {}
            data = self.ConvertToInput(kwargs.pop("data"))
            starts = self.ConvertToInput(kwargs.pop("starts"), dtype=np.int64)
            ends = self.ConvertToInput(kwargs.pop("ends"), dtype=np.int64)
            axes = self.ConvertToInput(
                kwargs.pop("axes", None), is_optional=True, dtype=np.int64
            )
            steps = self.ConvertToInput(
                kwargs.pop("steps", None), is_optional=True, dtype=np.int64
            )
            inputs = [data, starts, ends, axes, steps]

        # pro-process inputs and attr
        if kwargs:
            logger.warning("kwargs contains un-used key")

        new_attr = {}
        for key, val in attr.items():
            if val is not None:
                new_attr[key] = val
        attr = new_attr

        for ind, val in enumerate(inputs):
            if val is None:
                # empty string means no connection in ONNX
                inputs[ind] = util.ONNX_EMPTY_INPUT
        # remove tailing ""
        while inputs[-1] == util.ONNX_EMPTY_INPUT:
            inputs = inputs[:-1]

        if self.graph.opset >= 10:
            dtype = self.graph.get_dtype(inputs[1])
            for input_data in inputs[1:]:
                if input_data != util.ONNX_EMPTY_INPUT:
                    util.MakeSure(
                        dtype == self.graph.get_dtype(input_data),
                        "dtype should be same",
                    )

        return self.graph.MakeNode(
            op_type="Slice",
            inputs=inputs,
            attr=attr,
            name=name,
            outputs=outputs,
            shapes=shapes,
            dtypes=dtypes,
        ).output[0]

    def ConvertToInput(self, tensor, is_optional=False, dtype=None):
        """in ONNX, input shold come from node, so it must be a string"""
        if is_optional and tensor is None:
            return None

        util.MakeSure(tensor is not None, "input is required so it couldn't be None")

        res = tensor
        if isinstance(tensor, list):
            res = self.graph.MakeConst(
                id_util.UniqueStr("const_slice"), np.array(tensor, dtype)
            ).output[0]

        util.MakeSure(
            isinstance(res, str), "input is a dynamic input, so a str is needed"
        )

        return res

    def ConvertToAttribute(self, tensor, is_optional=False):
        if is_optional and tensor is None:
            return None

        util.MakeSure(tensor is not None, "input is required so it couldn't be None")

        res = tensor
        if isinstance(tensor, str):
            const_node = self.graph.get_node_by_output(tensor)
            res = const_node.get_tensor_value(as_list=True)

        util.MakeSure(isinstance(res, list), "input is an attr, so a list is needed")

        return res
