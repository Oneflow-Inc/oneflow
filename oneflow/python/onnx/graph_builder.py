# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.graph_helper - class to help building graph, such as helping to make complex node
"""

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

    def make_slice(self, kwargs, name=None, shapes=None, dtypes=None):
        """
        slice changes its schema at opset 10: it treats some attributes as dynamic input
        so this function has to process inputs according to graph's opset version
        to get "inputs" and "attr" to feed "make_node"
        kwargs: key could be ["data", "starts", "ends", "axes", "steps", "outputs"].
        """
        outputs = kwargs.pop("outputs", None)

        if self.graph.opset < 10:
            # "data" is string
            # "starts", "ends" and "axes" are attributes, and "axes" is optional.
            inputs = [kwargs.pop("data")]
            starts = self.convert_to_attribute(kwargs.pop("starts"))
            ends = self.convert_to_attribute(kwargs.pop("ends"))
            axes = self.convert_to_attribute(kwargs.pop("axes", None), is_optional=True)
            attr = {"starts": starts, "ends": ends, "axes": axes}
        else:
            # slice-10 has 3 required inputs "data", "starts", "ends"l
            # and 2 optional inputs "axes", "steps"
            # input sequence should be "data", "starts", "ends", "axes", "steps"
            attr = {}
            data = self.convert_to_input(kwargs.pop("data"))
            starts = self.convert_to_input(kwargs.pop("starts"), dtype=np.int64)
            ends = self.convert_to_input(kwargs.pop("ends"), dtype=np.int64)
            axes = self.convert_to_input(
                kwargs.pop("axes", None), is_optional=True, dtype=np.int64
            )
            steps = self.convert_to_input(
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
                    util.make_sure(
                        dtype == self.graph.get_dtype(input_data),
                        "dtype should be same",
                    )

        return self.graph.make_node(
            op_type="Slice",
            inputs=inputs,
            attr=attr,
            name=name,
            outputs=outputs,
            shapes=shapes,
            dtypes=dtypes,
        ).output[0]

    def convert_to_input(self, tensor, is_optional=False, dtype=None):
        """in ONNX, input shold come from node, so it must be a string"""
        if is_optional and tensor is None:
            return None

        util.make_sure(tensor is not None, "input is required so it couldn't be None")

        res = tensor
        if isinstance(tensor, list):
            res = self.graph.make_const(
                id_util.UniqueStr("const_slice"), np.array(tensor, dtype)
            ).output[0]

        util.make_sure(
            isinstance(res, str), "input is a dynamic input, so a str is needed"
        )

        return res

    def convert_to_attribute(self, tensor, is_optional=False):
        if is_optional and tensor is None:
            return None

        util.make_sure(tensor is not None, "input is required so it couldn't be None")

        res = tensor
        if isinstance(tensor, str):
            const_node = self.graph.get_node_by_output(tensor)
            res = const_node.get_tensor_value(as_list=True)

        util.make_sure(isinstance(res, list), "input is an attr, so a list is needed")

        return res
