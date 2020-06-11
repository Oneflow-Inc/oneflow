# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.graph - class to manage graph manipulation on top of onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import copy
import logging
import six
import numpy as np
from os.path import join as pathjoin

from onnx import helper, numpy_helper, shape_inference, OperatorSetIdProto, AttributeProto, TensorProto, onnx_pb
from oneflow.python.onnx import utils
from oneflow.python.onnx.utils import make_name, port_name, find_opset
from oneflow.python.onnx import optimizer
from oneflow.python.onnx.schemas import get_schema, infer_onnx_shape_dtype
from oneflow.python.onnx import constants

logger = logging.getLogger(__name__)


# todo(pengwa): remove protected-access later
# pylint: disable=broad-except,protected-access


class Node(object):
    """A Node - wrapper around onnx nodes that we use for graph manipulations."""

    def __init__(self, node, graph, skip_conversion=False):
        """Create Node.
        Args:
            node: Onnx node in NodeProto
            graph: Graph() we are part of
        """
        self._op = node
        self.graph = graph
        self._input = list(node.input)
        self._output = list(node.output)
        self._attr = {}

        graph.set_node_by_name(self)
        # dict to original attributes
        for a in node.attribute:
            self._attr[a.name] = a
        self._skip_conversion = skip_conversion

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, val):
        self._input = copy.deepcopy(val)

    @property
    def output(self):
        return copy.deepcopy(self._output)

    @output.setter
    def output(self, val):
        """Set op output. Output should be updated explicitly,
        changing it would require output mapping changed.
        """
        self._graph_check()
        for o in self._output:
            del self.graph._output_to_node_name[o]

        self._output = val
        for o in self._output:
            utils.make_sure(o not in self.graph._output_to_node_name,
                            "output %s already in output mapping", o)
            self.graph._output_to_node_name[o] = self.name

    @property
    def inputs(self):
        """Input node objects."""
        self._graph_check()
        val = [self.graph.get_node_by_output(n) for n in self._input]
        return val

    @property
    def attr(self):
        return self._attr

    @property
    def attr_onnx(self):
        """Return onnx valid attributes"""
        schema = get_schema(self.type, self.graph.opset, self.domain)
        if schema is None and not (self.is_const() or self.is_graph_input()):
            logger.debug("Node %s uses non-stardard onnx op <%s, %s>, skip attribute check",
                         self.name, self.domain, self.type)
        onnx_attrs = {}
        for a in self._attr.values():
            if schema is None or schema.has_attribute(a.name):
                onnx_attrs[a.name] = a
        return onnx_attrs

    @property
    def name(self):
        return self._op.name

    @property
    def op(self):
        """TODO: have a better interface for this."""
        return self._op

    @property
    def type(self):
        """Return Op type."""
        return self._op.op_type

    @type.setter
    def type(self, val):
        """Set Op type."""
        self._op.op_type = val

    @property
    def domain(self):
        """Return Op type."""
        return self._op.domain

    @domain.setter
    def domain(self, val):
        """Set Op type."""
        self._op.domain = val

    @property
    def data_format(self):
        """Return data_format."""
        return self.get_attr_str("data_format")

    @data_format.setter
    def data_format(self, val):
        """Set data_format."""
        self.set_attr("data_format", val)

    def is_nhwc(self):
        """Return True if node is in NHWC format."""
        if self.type == 'BatchNormalization':
            axis = self.get_attr_value('axis')
            return axis == -1 or axis == len(self.output_shapes[0]) - 1
        return self.data_format in ["NHWC", 'channels_last']

    def is_const(self):
        """Return True if node is a constant."""
        return self.type in ['variable', 'Const']
        # return self.type in ["Const", "ConstV2"]

    def is_graph_output(self):
        return self.type in ['return']

    def is_graph_input(self):
        return self.type in ['input']

    def is_graph_input_default_const(self):
        return self.is_const() and any(
            out.is_graph_input() for out in self.graph.find_output_consumers(self.output[0])
        )

    def __str__(self):
        return str(self._op)

    def __repr__(self):
        return "<onnx op type='%s' name=%s>" % (self.type, self._op.name)

    @property
    def summary(self):
        """Return node summary information."""
        lines = []
        lines.append("OP={}".format(self.type))
        lines.append("Name={}".format(self.name))

        g = self.graph
        if self.input:
            lines.append("Inputs:")
            for name in self.input:
                node = g.get_node_by_output(name)
                op = node.type if node else "N/A"
                lines.append("\t{}={}, {}, {}".format(
                    name, op, g.get_shape(name), g.get_dtype(name)))

        if self.output:
            for name in self.output:
                lines.append("Outpus:")
                lines.append("\t{}={}, {}".format(
                    name, g.get_shape(name), g.get_dtype(name)))

        return '\n'.join(lines)

    def get_attr(self, name, default=None):
        """Get raw attribute value."""
        attr = self.attr.get(name, default)
        return attr

    def get_attr_value(self, name, default=None):
        attr = self.get_attr(name)
        if attr:
            return helper.get_attribute_value(attr)
        return default

    def get_attr_int(self, name):
        """Get attribute value as int."""
        attr_int = self.get_attr_value(name)
        utils.make_sure(
            attr_int is not None and isinstance(attr_int, int),
            "attribute %s is None", name
        )
        return attr_int

    def get_attr_str(self, name, encoding="utf-8"):
        """Get attribute value as string."""
        attr_str = self.get_attr_value(name)
        utils.make_sure(
            attr_str is not None and isinstance(attr_str, bytes),
            "attribute %s is None", name
        )
        return attr_str.decode(encoding)

    def set_attr(self, name, value):
        self.attr[name] = helper.make_attribute(name, value)

    def set_attr_onnx(self, value):
        self.attr[value.name] = value

    @property
    def skip_conversion(self):
        return self._skip_conversion

    @skip_conversion.setter
    def skip_conversion(self, val):
        self._skip_conversion = val

    # If some Node is created as onnx_node, then we don't need convert it
    def need_skip(self):
        return self._skip_conversion

    @property
    def output_shapes(self):
        """Get output shapes."""
        self._graph_check()
        val = [self.graph.get_shape(n) for n in self._output]
        return val

    @property
    def output_dtypes(self):
        """Get output dtypes."""
        self._graph_check()
        val = [self.graph.get_dtype(n) for n in self._output]
        return val

    def get_tensor_value(self, as_list=True):
        """Get value for onnx tensor.
        Args:
            as_list: whether return numpy ndarray in list.
        Returns:
            If as_list=True, return the array as a (possibly nested) list.
            Otherwise, return data of type np.ndarray.

            If a tensor is a scalar having value 1,
                when as_list=False, return np.array(1), type is <class 'numpy.ndarray'>
                when as_list=True, return 1, type is <class 'int'>.
        """
        if not self.is_const():
            raise ValueError(
                "get tensor value: {} must be Const".format(self.name))
        if self.type == 'Const':
            t = self.get_attr("value")
            if t:
                t = numpy_helper.to_array(helper.get_attribute_value(t))
        else:
            self._graph_check()
            t = self.graph.get_saved_tensor(self)
        if as_list is True and t is not None:
            t = t.tolist()  # t might be scalar after tolist()
        return t

    def scalar_to_dim1(self):
        """Get value for onnx tensor."""
        if not self.is_const():
            raise ValueError(
                "get tensor value: {} must be Const".format(self.name))

        t = self.get_attr("value")
        if t:
            t = helper.get_attribute_value(t)
            if not t.dims:
                t.dims.extend([1])
        return t.dims

    def set_tensor_value(self, new_val):
        """Set new value for existing onnx tensor.
        Args:
            new_val: value of type numpy ndarray
        """
        if not self.is_const():
            raise ValueError(
                "set tensor value: {} must be Const".format(self.name))
        t = self.get_attr("value")
        if not t:
            raise ValueError("set tensor value: {} is None".format(self.name))
        t = helper.get_attribute_value(t)
        onnx_tensor = numpy_helper.from_array(new_val, t.name)
        del t
        self.set_attr("value", onnx_tensor)
        # track shapes in _output_shapes
        self._graph_check()
        self.graph.set_shape(onnx_tensor.name, onnx_tensor.dims)

    def get_body_graphs(self):
        self._graph_check()
        return self.graph.contained_graphs.get(self.name, None)

    def set_body_graph_as_attr(self, attr_name, graph):
        self._graph_check()
        if self.name not in self.graph.contained_graphs:
            self.graph.contained_graphs[self.name] = {}

        self.graph.contained_graphs[self.name].update({attr_name: graph})
        graph.parent_graph = self.graph

    def update_proto(self):
        """Update protobuf from internal structure."""
        nodes = list(self._op.input)
        for node in nodes:
            self._op.input.remove(node)
        self._op.input.extend(self.input)
        nodes = list(self._op.output)
        for node in nodes:
            self._op.output.remove(node)
        self._op.output.extend(self.output)

        # update attributes to proto
        del self._op.attribute[:]

        # check attribute of type GraphProto
        attr_graphs = self.get_body_graphs()
        if attr_graphs:
            for attr_name, sub_graph in attr_graphs.items():
                graph_proto = sub_graph.make_graph(
                    "graph for " + self.name + " " + attr_name)
                self.set_attr(attr_name, graph_proto)

        attr = list(self.attr_onnx.values())
        if attr:
            self._op.attribute.extend(attr)

    def get_implicit_inputs(self, recursive=True):
        """Get implicit inputs if the node has attributes being GraphProto."""
        output_available_in_cur_graph = set()
        all_node_inputs = set()

        graphs = []
        body_graphs = self.get_body_graphs()
        if body_graphs:
            graphs.extend(body_graphs.values())

        while graphs:
            graph = graphs.pop()
            for n in graph.get_nodes():
                output_available_in_cur_graph |= set(n.output)
                for i in n.input:
                    all_node_inputs.add(i)

                if recursive:
                    b_graphs = n.get_body_graphs()
                    if b_graphs:
                        graphs.extend(b_graphs.values())

        outer_scope_node_input_ids = all_node_inputs - output_available_in_cur_graph
        return list(outer_scope_node_input_ids)

    def _graph_check(self):
        utils.make_sure(self.graph is not None, "Node %s not belonging any graph",
                        self.name)

    def maybe_cast_input(self, supported, type_map):
        """.maybe_cast_input
        Args:
            supported: list of supported types for inputs
            type_map: dict type to supported type mapping
        """
        did_cast = False
        for i, name in enumerate(self.input):
            dtype = self.graph.get_dtype(name)
            if dtype not in supported[i]:
                tdtype = type_map.get(dtype)
                if tdtype is None:
                    raise RuntimeError(
                        "don't know how to cast type {} on node {}".format(dtype, name))
                shape = self.graph.get_shape(name)
                cast_node = self.graph.insert_new_node_on_input(
                    self, "Cast", name)
                cast_node.set_attr("to", tdtype)
                self.graph.set_dtype(cast_node.output[0], [tdtype])
                self.graph.set_shape(cast_node.output[0], shape)
                did_cast = True
        return did_cast


class Graph(object):
    """"Class that provides graph manipulation and matching."""

    def __init__(self, nodes, model_save_dir, output_shapes=None, dtypes=None, target=None, opset=None, extra_opset=None,
                 output_names=None, input_maps=None):
        """Create Graph.
        Args:
            nodes: list of Node()
            output_shapes: dict of oneflow output shapes
            dtypes: dict of oneflow dtype
            input_maps: map (node_name, key) to value_names
        """
        if target is None:
            target = []
        self._nodes = []
        self._nodes_by_name = {}
        self._output_to_node_name = {}
        self.shapes = {}

        self._target = set(target)
        self._dtypes = dtypes

        self._model_save_dir = model_save_dir
        self._output_shapes = output_shapes
        self._opset = find_opset(opset)

        if extra_opset is not None:
            utils.make_sure(isinstance(extra_opset, list),
                            "invalid extra_opset")
        self._extra_opset = extra_opset

        self._order_sensitive_inputs = []
        self.outputs = output_names if output_names is not None else []
        self._input_maps = input_maps if input_maps is not None else {}

        self.parent_graph = None
        self.contained_graphs = {}  # {node_name: {node_attribute_name: Graph}}

        ops = [Node(node, self) for node in nodes]
        self.reset_nodes(ops)

        for op in ops:
            if op.is_graph_output():
                self.add_graph_output(op.input[0])

        # add identity node after each output, in case it is renamed during conversion.
        for o in self.outputs:
            n = self.get_node_by_output_in_current_graph(o)
            new_output_name = port_name(
                n.name + "_" + utils.make_name("raw_output_"))
            n_shapes = n.output_shapes
            n_dtypes = n.output_dtypes
            body_graphs = n.graph.contained_graphs.pop(n.name, None)
            self.remove_node(n.name)

            new_outputs = [output if output !=
                           o else new_output_name for output in n.output]
            # domain should be passed to new node
            new_node = self.make_node(n.type, n.input, outputs=new_outputs, attr=n.attr, name=n.name,
                                      skip_conversion=n._skip_conversion, dtypes=n_dtypes, shapes=n_shapes,
                                      domain=n.domain)

            if body_graphs:
                for attr_name, body_graph in body_graphs.items():
                    body_graph.parent_graph = self
                    new_node.set_body_graph_as_attr(attr_name, body_graph)

            self.replace_all_inputs(self.get_nodes(), o, new_output_name)
            self.make_node("Identity", [new_output_name], outputs=[
                           o], op_name_scope=n.name + "_" + "graph_outputs")
            self.copy_shape(new_output_name, o)
            self.copy_dtype(new_output_name, o)

    def create_new_graph_with_same_config(self):
        """Create a clean graph inheriting current graph's configuration."""
        return Graph([], output_shapes={}, dtypes={}, target=self._target, opset=self._opset,
                     extra_opset=self.extra_opset, output_names=[])

    @property
    def opset(self):
        return self._opset

    @property
    def extra_opset(self):
        return self._extra_opset

    def is_target(self, *names):
        """Return True if target platform contains any name."""
        return any(name in self._target for name in names)

    def make_const(self, name, np_val, skip_conversion=False, raw=True):
        """Make a new constant in the graph.
        Args:
            name: const node name, must be unique.
            np_val: value of type numpy ndarray.
            skip_conversion: bool, indicate whether this created node would be mapped during conversion.
            raw: whether to store data at field of raw_data or the specific field according to its dtype
        """
        if raw:
            onnx_tensor = numpy_helper.from_array(np_val, name)
        else:
            onnx_tensor = helper.make_tensor(name, utils.map_numpy_to_onnx_dtype(np_val.dtype),
                                             np_val.shape, np_val, raw=False)
        dtype = onnx_tensor.data_type
        node = self.make_node("Const", [], outputs=[name], name=name, attr={"value": onnx_tensor},
                              skip_conversion=skip_conversion, dtypes=[dtype], infer_shape_dtype=False)
        self.set_shape(name, np_val.shape)
        self.set_dtype(name, utils.map_numpy_to_onnx_dtype(np_val.dtype))
        return node

    def make_node(self, op_type, inputs, attr=None, output_count=1, outputs=None, skip_conversion=True,
                  op_name_scope=None, name=None, shapes=None, dtypes=None, domain=constants.ONNX_DOMAIN,
                  infer_shape_dtype=True):
        """Make a new onnx node in the graph"""
        if attr is None:
            attr = {}
        if shapes is None:
            shapes = []
        if dtypes is None:
            dtypes = []

        if name is None:
            name = utils.make_name(op_type)

        if op_name_scope:
            name = "_".join([op_name_scope, name])

        logger.debug("Making node: Name=%s, OP=%s", name, op_type)

        if outputs is None:
            outputs = [name + ":" + str(i) for i in range(output_count)]

        output_count = len(outputs)
        raw_attr = {}
        onnx_attrs = []
        for a, v in attr.items():
            if isinstance(v, AttributeProto):
                onnx_attrs.append(v)
            else:
                raw_attr[a] = v

        n = self.get_node_by_name(name)
        utils.make_sure(
            n is None, "name %s already exists in node: \n%s", name, n)
        for o in outputs:
            n = self.get_node_by_output_in_current_graph(o)
            utils.make_sure(
                n is None, "output tensor named %s already exists in node: \n%s", o, n)

        onnx_node = helper.make_node(
            op_type, inputs, outputs, name=name, domain=domain, **raw_attr)

        if op_type in ["If", "Loop", "Scan"]:
            # we force the op containing inner graphs not skipped during conversion.
            skip_conversion = False

        node = Node(onnx_node, self, skip_conversion=skip_conversion)
        if onnx_attrs:
            _ = [node.set_attr_onnx(a) for a in onnx_attrs]

        if shapes:
            utils.make_sure(len(shapes) == output_count,
                            "output shape count %s not equal to output count %s", len(shapes), output_count)
            for i in range(output_count):
                self.set_shape(node.output[i], shapes[i])

        if dtypes:
            utils.make_sure(len(dtypes) == output_count,
                            "output dtypes count %s not equal to output count %s", len(dtypes), output_count)
            for i in range(output_count):
                self.set_dtype(node.output[i], dtypes[i])

        if (not shapes or not dtypes) and infer_shape_dtype:
            self.update_node_shape_dtype(node, override=False)

        logger.debug("Made node: %s\n%s", node.name, node.summary)
        self._nodes.append(node)
        return node

    def remove_node(self, node_name):
        """Remove node in current graph."""
        utils.make_sure(node_name in self._nodes_by_name,
                        "node %s not in current graph, cannot remove", node_name)
        node = self.get_node_by_name(node_name)
        del self._nodes_by_name[node_name]
        if node_name in self.contained_graphs:
            del self.contained_graphs[node_name]

        if node in self._order_sensitive_inputs:
            self._order_sensitive_inputs.remove(node)

        for op_output in node.output:
            del self._output_to_node_name[op_output]

            if op_output in self._output_shapes:
                del self._output_shapes[op_output]
            if op_output in self._dtypes:
                del self._dtypes[op_output]

        self._nodes.remove(node)
        node.graph = None

    def get_inputs(self, node, key):
        return self._input_maps[node.name][key]

    def reset_nodes(self, ops):
        """Reset the graph with node list."""
        remained_dtypes = {}
        remained_shapes = {}
        remained_sub_graphs = {}
        for op in ops:
            for op_output in op.output:
                # this check should be removed once we make sure all output tensors have dtype/shape.
                if op_output in self._dtypes:
                    remained_dtypes[op_output] = self._dtypes[op_output]
                if op_output in self._output_shapes:
                    remained_shapes[op_output] = self._output_shapes[op_output]

            if op.name in self.contained_graphs:
                remained_sub_graphs[op.name] = self.contained_graphs[op.name]

        self._nodes = ops
        self.contained_graphs = remained_sub_graphs
        self._nodes_by_name = {op.name: op for op in ops}
        self._output_to_node_name = {}
        for op in ops:
            for op_output in op.output:
                self._output_to_node_name[op_output] = op.name

        for n in self._order_sensitive_inputs:
            if n not in ops:
                self._order_sensitive_inputs.remove(n)
        for o in self.outputs:
            if o not in self._output_to_node_name:
                raise ValueError("graph output " + o + " not exist")

        self._dtypes = remained_dtypes
        self._output_shapes = remained_shapes

    def is_empty_input(self, name):
        # in ONNX, operation may have optional input and an empty string may be used
        # in the place of an actual argument's name to indicate a missing argument
        return name == utils.ONNX_EMPTY_INPUT

    def check_integrity(self):
        """
        Check graph integrity. Every node's input needs to associate with a node.
        Return broken outputs.
        """
        broken_outputs = set()
        for node in self.get_nodes():
            for inp in node.input:
                if self.get_node_by_output(inp) is None and not self.is_empty_input(inp):
                    broken_outputs.add(inp)
        return list(broken_outputs)

    def update_node_shape_dtype(self, node, override=False):
        """Try the best to infer shapes and dtypes for outputs of the node,
        by default, we respect TF shapes and dtypes.
        """
        if node.is_const() or node.is_graph_input():
            return
        # NOTE: only support onnx node for now
        if not utils.is_onnx_domain(node.domain):
            return

        logger.debug("Infer shape and dtype for [%s]", node.name)
        # NOTE: shape inference for some ops need the input values of the op, e.g., Reshape
        # op needs the "Shape" value to infer output shape.
        initializers = []
        for i, inp in enumerate(node.inputs):
            if inp is None:
                if not self.is_empty_input(node.input[i]):
                    if logger.isEnabledFor(logging.INFO):
                        logger.warning(
                            "[%s] infer a inexistent node: [%s], please check the code",
                            node.name, node.input[i]
                        )
                continue
            if inp.is_const():
                tensor = numpy_helper.from_array(
                    inp.get_tensor_value(as_list=False), name=inp.output[0])
                initializers.append(tensor)

        input_shapes = [self.get_shape(i) for i in node.input]
        input_dtypes = [self.get_dtype(i) for i in node.input]

        shapes, dtypes = infer_onnx_shape_dtype(
            node, self._opset, input_shapes, input_dtypes, initializers)
        if not shapes or not dtypes:
            return

        for output, shape, dtype in zip(node.output, shapes, dtypes):
            if dtype == TensorProto.UNDEFINED:
                logger.debug(
                    "Inferred dtype for [%s, type: %s] is UNDEFINED, SKIP", node.name, node.type)
            else:
                existing_dtype = self.get_dtype(output)
                if existing_dtype is not None and existing_dtype != dtype:
                    if override:
                        logger.warning(
                            "Override dtype of %s from %s to %s", output, existing_dtype, dtype)
                    else:
                        dtype = existing_dtype
                self.set_dtype(output, dtype)
                logger.debug("Set dtype of [%s] to %s", output, dtype)

            if shape is None:
                logger.debug(
                    "Inferred shape for [%s, type: %s] is None, SKIP", node.name, node.type)
            else:
                existing_shape = self.get_shape(output)
                if existing_shape is not None and not utils.are_shapes_equal(existing_shape, shape):
                    if override:
                        logger.warning(
                            "Override shape of %s from %s to %s", output, existing_shape, shape)
                    else:
                        shape = existing_shape
                self.set_shape(output, shape)
                logger.debug("Set shape of [%s] to %s", output, shape)

    def update_proto(self):
        """Update the onnx protobuf from out internal Node structure."""
        for node in self._nodes:
            node.update_proto()

    def get_nodes(self):
        """Get node list."""
        return self._nodes

    def get_node_by_output(self, output, search_in_parent_graphs=True):
        """Get node by node output id recursively going through nested graphs.
        Args:
            search_in_parent_graphs: search in all parent graphs
        """
        ret = None
        g = self
        while not ret and g:
            ret = g.get_node_by_output_in_current_graph(output)
            if ret:
                return ret

            if not search_in_parent_graphs:
                break
            g = g.parent_graph
        return ret

    def get_node_by_output_in_current_graph(self, output):
        """Get node by node output id."""
        name = self._output_to_node_name.get(output)
        ret = None
        if name:
            ret = self._nodes_by_name.get(name)
        return ret

    def get_node_by_name(self, name):
        """Get node by name."""
        ret = self._nodes_by_name.get(name)
        return ret

    def set_node_by_name(self, node):
        """Set node by name."""
        self._nodes_by_name[node.name] = node
        for op_output in node.output:
            self._output_to_node_name[op_output] = node.name

    def add_graph_input(self, name, dtype=None, shape=None):
        """Add placeholder node as graph's input. Order matters only for subgraph.
           Placeholders in original graph are assumed for main graph, order not matters.
        """
        if dtype is None:
            dtype = self.get_dtype(name)

        if shape is None:
            shape = self.get_shape(name)

        new_node = self.make_node("Placeholder", [], outputs=[
                                  name], dtypes=[dtype], shapes=[shape])
        self._order_sensitive_inputs.append(new_node)

    def add_graph_input_with_default(self, name, default_const, dtype=None, shape=None):
        """Add placeholderwithdefault."""
        if dtype is None:
            dtype = self.get_dtype(name)

        if shape is None:
            shape = self.get_shape(name)

        default_const_name = port_name(make_name("{}_default".format(name)))
        default_const.output = [default_const_name]
        new_node = self.make_node("PlaceholderWithDefault", [default_const_name], outputs=[name],
                                  dtypes=[dtype], shapes=[shape])
        self._order_sensitive_inputs.append(new_node)

    def add_graph_output(self, name, dtype=None, shape=None):
        """Add node output as graph's output."""
        utils.make_sure(name in self._output_to_node_name,
                        "output %s not exist in the graph", name)

        if dtype is None:
            dtype = self.get_dtype(name)

        if shape is None:
            shape = self.get_shape(name)

        if name not in self.outputs:
            utils.make_sure(shape is not None,
                            "shape for output %s should not be None", name)
            utils.make_sure(dtype is not None,
                            "dtype for output %s should not be None", name)
            self.outputs.append(name)
            self.set_shape(name, shape)
            self.set_dtype(name, dtype)
        else:
            raise ValueError("graph output " + name + " already exists")

    def get_dtype(self, name):
        """Get dtype for node."""
        node = self.get_node_by_output(name, search_in_parent_graphs=True)
        return node.graph._dtypes.get(name) if node else None

    def set_dtype(self, name, dtype):
        """Set dtype for node."""
        node = self.get_node_by_output(name, search_in_parent_graphs=True)
        node.graph._dtypes[name] = dtype

    def copy_dtype(self, src_name, dst_name):
        """Copy dtype from another node."""
        dtype = self.get_dtype(src_name)
        self.set_dtype(dst_name, dtype)

    def get_saved_tensor(self, node):
        # TODO(daquexian): node.output[0] is "node_name/output_name", so pathjoin is not cross-platform
        tensor_name = node.output[0]
        if self._model_save_dir is not None:
            path = pathjoin(self._model_save_dir, node.output[0])
            tensor_value = np.fromfile(path, dtype=utils.map_onnx_to_numpy_type(
                self.get_dtype(tensor_name))).reshape(self.get_shape(tensor_name))
            return tensor_value
        else:
            shape = self.get_shape(tensor_name)
            return np.ones(shape, dtype=utils.map_onnx_to_numpy_type(self.get_dtype(tensor_name)))

    def get_shape(self, name):
        """Get shape for node."""
        utils.make_sure(isinstance(name, six.text_type),
                        "get_shape name is invalid type: %s", name)
        node = self.get_node_by_output(name, search_in_parent_graphs=True)
        shape = node.graph._output_shapes.get(name) if node else None
        if shape:
            for i, v in enumerate(shape):
                if v is None:
                    # pylint: disable=unsupported-assignment-operation
                    shape[i] = -1
            # hack to allow utils.ONNX_UNKNOWN_DIMENSION to override batchsize if needed.
            # default is -1.
            if shape[0] == -1:
                # pylint: disable=unsupported-assignment-operation
                shape[0] = utils.ONNX_UNKNOWN_DIMENSION
            return shape
        return shape

    def set_shape(self, name, val):
        """Set new shape of node."""
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, tuple):
            val = list(val)
        node = self.get_node_by_output(name, search_in_parent_graphs=True)
        utils.make_sure(node is not None,
                        "cannot find node by output id %s", name)
        node.graph._output_shapes[name] = val

    def copy_shape(self, input_name, output_name):
        """Copy shape from another node."""
        shape = self.get_shape(input_name)
        # assert shape is not None
        if shape is not None:
            self.set_shape(output_name, shape)

    def topological_sort(self, ops):
        """Topological sort of graph."""
        # sort by name, the result will be reversed alphabeta
        ops.sort(key=lambda op: op.name)

        def _push_stack(stack, node, in_stack):
            stack.append(node)
            if node in in_stack:
                raise ValueError('Graph has cycles.')
            in_stack[node] = True

        def _get_unvisited_child(g, node, not_visited):
            for child in g[node]:
                if child in not_visited:
                    return child
            return -1

        n = len(ops)
        g = [[] for _ in range(n)]
        op_name_to_index = {}
        for i, op in enumerate(ops):
            op_name_to_index[op.name] = i

        for i, op in enumerate(ops):
            all_input = set(op.input)
            implicit_inputs = op.get_implicit_inputs()
            all_input |= set(implicit_inputs)
            # remove those empty inputs
            all_input = list(filter(lambda a: a != '', all_input))
            for inp in sorted(all_input):
                j = self.get_node_by_output(inp)
                utils.make_sure(
                    j is not None, "Cannot find node with output {}".format(inp))
                if self.parent_graph and j.name not in op_name_to_index:
                    # there might be some outer-scoped inputs for an inner Graph.
                    pass
                else:
                    g[op_name_to_index[j.name]].append(i)

        # label for each op. highest = sink nodes.
        label = [-1 for _ in range(n)]
        stack = []
        in_stack = dict()
        not_visited = dict.fromkeys(range(n))
        label_counter = n - 1

        while not_visited:
            node = list(not_visited.keys())[0]
            _push_stack(stack, node, in_stack)
            while stack:
                node = _get_unvisited_child(g, stack[-1], not_visited)
                if node != -1:
                    _push_stack(stack, node, in_stack)
                else:
                    node = stack.pop()
                    in_stack.pop(node)
                    not_visited.pop(node)
                    label[node] = label_counter
                    label_counter -= 1

        ret = [x for _, x in sorted(zip(label, ops))]
        self.reset_nodes(ret)

    def make_graph(self, doc, graph_name="oneflow.python.onnx"):
        """
        Create GraphProto for onnx from internal graph.
        Args:
            optimize: optimize graph via onnx
            doc: text for doc string of the graph
        """
        self.delete_unused_nodes(self.outputs)
        self.topological_sort(self.get_nodes())
        self.update_proto()

        # TODO: we'd want to do something like this so that transpose optimizer is active
        # for  all (unit) tests
        # if optimize:
        #    from oneflow.python.onnx.optimizer.transpose_optimizer import TransposeOptimizer
        #    optimizer = TransposeOptimizer(self, False)
        #    optimizer.optimize()
        ops = []
        order_non_sensitive_placeholders = []
        order_sensitive_placeholders = self._order_sensitive_inputs
        const_ops = []
        output_ops = []
        for op in self.get_nodes():
            if op.is_const():
                const_ops.append(op)
                continue
            if op.is_graph_input():
                if op not in self._order_sensitive_inputs:
                    order_non_sensitive_placeholders.append(op)
                continue
            ops.append(op)
        placeholder_ops = order_sensitive_placeholders + order_non_sensitive_placeholders

        # create initializers for placeholder with default nodes
        initializers = []
        placeholder_default_const_ops = []
        for op in placeholder_ops:
            if op.type == "PlaceholderWithDefault":
                utils.make_sure(
                    op.inputs[0] is not None, "Cannot find node with output {}".format(op.input[0]))
                utils.make_sure(op.inputs[0].is_const(),
                                "non-const default value for PlaceholderWithDefault is not supported.")
                # copy the tensor value, set its name to current node's output, add as initializer
                value = op.inputs[0].get_tensor_value(as_list=False)
                tensor = numpy_helper.from_array(value, op.output[0])
                initializers.append(tensor)
                placeholder_default_const_ops.append(op.inputs[0])

        # create initializers for constant nodes
        const_ops = [
            op for op in const_ops if op not in placeholder_default_const_ops]
        for op in const_ops:
            tensor_name = op.output[0]
            tensor = numpy_helper.from_array(
                op.get_tensor_value(as_list=False), tensor_name)
            initializers.append(tensor)

        # create input_tensor_values
        input_ids = [op.output[0] for op in placeholder_ops]
        # onnx with IR version below 4 requires initializer should be in inputs.
        # here we check opset version rather than IR version for the reason:
        # https://github.com/onnx/tensorflow-onnx/pull/557
        # opset 9 come with IR 4.
        if self.opset < 9:
            input_ids += [op.output[0] for op in const_ops]

        input_tensor_values = self.make_onnx_graph_io(input_ids)

        # create output_tensor_values
        output_tensor_values = self.make_onnx_graph_io(self.outputs)

        # create graph proto
        graph = helper.make_graph([op.op for op in ops],
                                  graph_name,
                                  input_tensor_values,
                                  output_tensor_values,
                                  initializer=initializers,
                                  doc_string=doc)

        return graph

    def make_model(self, graph_doc, optimize=False, graph_name="oneflow.python.onnx", **kwargs):
        """
        Create final ModelProto for onnx from internal graph.
        Args:
            optimize: optimize graph via onnx
            doc: text for doc string of the model
        """
        graph = self.make_graph(graph_doc, graph_name)

        if "producer_name" not in kwargs:
            kwargs = {"producer_name": "oneflow.python.onnx"}

        if "opset_imports" not in kwargs:
            opsets = []
            imp = OperatorSetIdProto()
            imp.version = self._opset
            opsets.append(imp)
            if self.extra_opset is not None:
                opsets.extend(self.extra_opset)
            kwargs["opset_imports"] = opsets
        model_proto = helper.make_model(graph, **kwargs)

        # optimize the model proto.
        # TODO: this is disabled by default because of bugs in fuse_consecutive_transposes
        if optimize:
            model_proto = optimizer.optimize(model_proto)
        return model_proto

    def make_onnx_graph_io(self, ids):
        """Create tensor_value_info for passed input/output ids."""
        tensor_value_infos = []
        for name in ids:
            dtype = self.get_dtype(name)
            shape = self.get_shape(name)

            utils.make_sure(dtype is not None,
                            "missing output dtype for " + name)
            utils.make_sure(shape is not None,
                            "missing output shape for " + name)

            v = utils.make_onnx_inputs_outputs(name, dtype, shape)
            tensor_value_infos.append(v)
        return tensor_value_infos

    def dump_graph(self):
        """Dump graph with shapes (helpful for debugging)."""
        for node in self.get_nodes():
            input_names = ["{}{}".format(n, self.get_shape(n))
                           for n in node.input]
            logger.debug("%s %s %s %s",
                         node.type,
                         self.get_shape(node.output[0]),
                         node.name,
                         ", ".join(input_names))

    def follow_inputs(self, node, num, space=""):
        """Follow inputs for (helpful for debugging)."""
        val = []
        top = space == ""
        if num == 0:
            return []
        val.append("{}{} {} {}".format(space, node.type, node.name,
                                       self.get_shape(port_name(node.name))))
        space += "    "
        for j in node.inputs:
            val.extend(self.follow_inputs(j, num - 1, space))
        if top:
            print("\n".join(reversed(val)))
            print()
            return []
        return val

    def dump_node_statistics(self):
        op_cnt = collections.Counter()
        for n in self.get_nodes():
            op_cnt[n.type] += 1
            body_graphs = n.get_body_graphs()
            if body_graphs:
                for _, b_g in body_graphs.items():
                    op_cnt += b_g.dump_node_statistics()

        return op_cnt

    @staticmethod
    def remove_input(node, to_be_removed):
        """Remove input from Node.
        Args:
            node: the node we expect the input on
            to_be_removed: the node name we want to remove
        """
        assert isinstance(node, Node) and isinstance(
            to_be_removed, six.text_type)
        for i, name in enumerate(node.input):
            if name == to_be_removed:
                del node.input[i]
                break
        # don't remove output from parent since others might depend on it
        return True

    def insert_new_node_on_input(self, node, op_type, input_name, name=None, domain=None, **kwargs):
        """Create and insert a new node into the graph.
        Args:
            node: we want to replace the input for this node
            op_type: type for new operation
            input_name: the name(s) of the outputs above us
                if scalar, new node placed above input_name
                if list, new node placed above input_name[0]. list is inputs into new node
            name: the name of the new op
            kwargs: attributes of the new node

        Returns:
            node that was inserted
        """
        if name is None:
            name = utils.make_name(node.name)
        new_output = port_name(name)
        if not isinstance(input_name, list):
            input_name = [input_name]

        new_node = self.make_node(op_type, input_name, attr=kwargs, outputs=[
                                  new_output], name=name, domain=domain)
        for i, n in enumerate(node.input):
            if n == input_name[0]:
                node.input[i] = new_output
                break
        return new_node

    def insert_new_node_on_output(self, op_type, output_name, name, domain=None, **kwargs):
        """Create and insert a new node into the graph.
        Args:
            op_type: type for new operation
            output_name: the names of the outputs above us
            name: the name of the new op
            kwargs: attributes of the new node

        Returns:
            node that was inserted
        """
        utils.make_sure(isinstance(output_name, six.text_type), "output_name's type is not expected: %s",
                        type(output_name))
        utils.make_sure(isinstance(op_type, six.text_type), "op_type's type is not expected: %s",
                        type(op_type))

        new_output = port_name(name)
        new_node = self.make_node(op_type, [output_name], attr=kwargs, outputs=[
                                  new_output], name=name, domain=domain)

        to_replace = [n for n in self.get_nodes() if n != new_node]
        self.replace_all_inputs(to_replace, output_name, new_output)
        return new_node

    def find_output_consumers(self, output_name):
        """Find all nodes consuming a given output."""
        nodes = []
        for node in self.get_nodes():
            if output_name in node.input:
                nodes.append(node)

            # find consumers in sub graphs
            body_graphs = node.get_body_graphs()
            if body_graphs:
                for g in body_graphs.values():
                    nodes.extend(g.find_output_consumers(output_name))
        return nodes

    @staticmethod
    def replace_all_inputs(ops, old_input, new_input):
        """Replace all inputs pointing to old_input with new_input."""
        if old_input == new_input:
            return

        for node in ops:
            if old_input in node.input and new_input in node.output:
                raise RuntimeError(
                    "creating a circle in the graph is not allowed: " + node.name)

            for i, input_name in enumerate(node.input):
                if input_name == old_input:
                    node.input[i] = new_input
                    if node.name in node.graph._input_maps:
                        for k, v in node.graph._input_maps[node.name].items():
                            assert len(v) == 1
                            if v[0] == old_input:
                                node.graph._input_maps[node.name][k] = [
                                    new_input]

            # modify references in sub graphs
            body_graphs = node.get_body_graphs()
            if body_graphs:
                for g in body_graphs.values():
                    g.replace_all_inputs(g.get_nodes(), old_input, new_input)

    @staticmethod
    def replace_input(node, old_input, new_input):
        """Replace node."""
        assert isinstance(node, Node) and isinstance(
            old_input, six.text_type) and isinstance(new_input, six.text_type)
        is_replaced = False
        for i, input_name in enumerate(node.input):
            if input_name == old_input:
                node.input[i] = new_input
                if node.name in node.graph._input_maps:
                    for k, v in node.graph._input_maps[node.name].items():
                        assert len(v) == 1
                        if v[0] == old_input:
                            node.graph._input_maps[node.name][k] = [new_input]
                is_replaced = True
        return is_replaced

    def _extract_sub_graph_nodes(self, dest_node, input_checker=None):
        """Return nodes of subgraph ending with dest_node.
        Args:
            dest_node: output node of the subgraph to find
            input_checker: customized input check function: bool func(node)

        Return:
            a set of nodes
        """
        res_set = set()
        if not dest_node or (input_checker and input_checker(dest_node) is False):
            return res_set

        processing_set = set([dest_node])
        while processing_set:
            top_node = processing_set.pop()
            res_set.add(top_node)
            all_inputs = top_node.input + list(top_node.get_implicit_inputs())
            for input_id in all_inputs:
                # we don't care about nested graph here, just handle current graph cropping.
                node = self.get_node_by_output(
                    input_id, search_in_parent_graphs=False)
                if not node:
                    # some nodes (for example Scan) have optional inputs, which
                    # might have empty input.
                    # subgraph might have input defined in outer graph
                    continue
                if node not in res_set:
                    if input_checker and input_checker(node) is False:
                        continue
                    processing_set.add(node)
        return res_set

    def extract_sub_graph_nodes(self, outputs_name, input_checker=None, ignore_unused_placeholder=True):
        """Return nodes of subgraph having output_ids as outputs.
        Args:
            output_ids: output node output id of the subgraph to find
            input_checker: customized input check function: bool func(node)
            ignore_unused_placeholder: bool, indicates whether unused placeholder will be removed
                in the resulting nodes.
        Return:
            a list of nodes
        """
        res_set = set()
        if not outputs_name:
            return list(res_set)

        for output in outputs_name:
            node = self.get_node_by_output(
                output, search_in_parent_graphs=False)
            res_set = res_set.union(
                self._extract_sub_graph_nodes(node, input_checker))

        if not ignore_unused_placeholder:
            # add back placeholder nodes if they are not connected to outputs.
            for node in self.get_nodes():
                if node.is_graph_input():
                    if node not in res_set:
                        res_set.add(node)
                    if node.type == "PlaceholderWithDefault" and \
                            node.inputs[0].is_const() and \
                            node.inputs[0] not in res_set:
                        res_set.add(node.inputs[0])

        return list(res_set)

    def delete_unused_nodes(self, outputs_name):
        """Delete nodes not in subgraph ending with output_names."""
        if not outputs_name:
            logger.debug(
                "Outputs not specified, delete_unused_nodes not taking effect.")
            return

        # we need keep those placeholders that are used as input of Loop's body graph.
        # some of them are not used in the graph, but still need be there to keep the graph complete.
        related_nodes = self.extract_sub_graph_nodes(
            outputs_name, ignore_unused_placeholder=False)
        for node in related_nodes:
            attr_body_graphs = node.get_body_graphs()
            if attr_body_graphs:
                for _, body_graph in attr_body_graphs.items():
                    body_graph.delete_unused_nodes(body_graph.outputs)
        self.reset_nodes(related_nodes)

    def safe_to_remove_nodes(self, to_delete):
        """ List of nodes that safe to delete (i.e. outputs not consumed by other nodes.)"""
        safe_to_remove = []
        delete_set = set(to_delete)
        for n in delete_set:
            out_consumers = set()
            for out in n.output:
                out_consumers |= set(self.find_output_consumers(out))
            if out_consumers.issubset(delete_set):
                safe_to_remove.append(n)
        return safe_to_remove

    def safe_remove_nodes(self, to_delete):
        """Delete nodes in `to_delete` without third-party node consuming it."""
        delete_set = set(to_delete)
        for n in delete_set:
            out_consumers = set()
            for out in n.output:
                out_consumers |= set(self.find_output_consumers(out))
            if out_consumers.issubset(delete_set):
                self.remove_node(n.name)


class GraphUtil(object):
    """Utilities for Graph manipulation."""

    @staticmethod
    def optimize_graph(graph):
        return optimizer.optimize_graph(graph)

    @staticmethod
    def optimize_model_proto(onnx_model_proto):
        """Optimize the model proto, for example: eliminating all useless Transpose pairs.

        Returns:
            model proto after optimization, if optimizer run successfully
            or onnx_model_proto, if exceptions happens
        """
        try:
            kwargs = GraphUtil.get_onnx_model_properties(onnx_model_proto)
            graph = GraphUtil.create_graph_from_onnx_model(onnx_model_proto)
            graph = GraphUtil.optimize_graph(graph)
            model_proto = graph.make_model(onnx_model_proto.graph.doc_string,
                                           graph_name=onnx_model_proto.graph.name, **kwargs)

            if onnx_model_proto.metadata_props:
                metadata_props = {
                    p.key: p.value for p in onnx_model_proto.metadata_props}
                helper.set_model_props(model_proto, metadata_props)
            return model_proto
        except Exception:
            # sometimes, onnx shape inference will fail for some reason,
            # return onnx_model_proto for this case
            logger.warning("Failed to optimize model proto", exc_info=1)
            return onnx_model_proto

    @staticmethod
    def get_onnx_model_properties(onnx_model_proto):
        """Get ModelProto properties"""
        kwargs = {}
        if onnx_model_proto.HasField('ir_version'):
            kwargs["ir_version"] = onnx_model_proto.ir_version
        if onnx_model_proto.HasField('producer_name'):
            kwargs["producer_name"] = onnx_model_proto.producer_name
        if onnx_model_proto.HasField('producer_version'):
            kwargs["producer_version"] = onnx_model_proto.producer_version
        if onnx_model_proto.HasField('domain'):
            kwargs["domain"] = onnx_model_proto.domain
        if onnx_model_proto.HasField('model_version'):
            kwargs["model_version"] = onnx_model_proto.model_version
        if onnx_model_proto.HasField('doc_string'):
            kwargs["doc_string"] = onnx_model_proto.doc_string
        kwargs["opset_imports"] = onnx_model_proto.opset_import

        return kwargs

    @staticmethod
    def create_graph_from_onnx_model(onnx_model_proto):
        """Create Graph loading onnx model proto."""
        # apply shape inference on the model
        inferred_model = shape_inference.infer_shapes(onnx_model_proto)
        graph_proto = inferred_model.graph

        opset_version = None
        extra_opset = []
        for opset in onnx_model_proto.opset_import:
            if not opset.domain:
                # domain field is None or empty means it is onnx domain
                opset_version = opset.version
            else:
                extra_opset.append(opset)

        utils.make_sure(opset_version is not None,
                        "opset version is not specified for onnx domain")
        main_graph = GraphUtil.create_graph_from_onnx_graph(
            graph_proto, opset_version, extra_opset)
        return main_graph

    @staticmethod
    def create_graph_from_onnx_graph(graph_proto, opset_version=None, extra_opset=None):
        """Create Graph loading onnx graph proto."""
        output_shapes = {}
        output_dtypes = {}

        shapes, dtypes = GraphUtil._parse_shape_and_type_from_value_infos(
            graph_proto.value_info)
        output_shapes.update(shapes)
        output_dtypes.update(dtypes)

        shapes, dtypes = GraphUtil._parse_shape_and_type_from_value_infos(
            graph_proto.output)
        output_shapes.update(shapes)
        output_dtypes.update(dtypes)

        nodes_to_append = []
        for n in graph_proto.node:
            if n.op_type == "Constant":
                n.op_type = "Const"

            # some pytorch model had empty names - make one up
            if not n.name:
                n.name = utils.make_name("was_empty")
            nodes_to_append.append(n)

        output_names = []
        for n in graph_proto.output:
            output_names.append(n.name)

        g = Graph(nodes_to_append, output_shapes, output_dtypes,
                  None, opset_version, extra_opset, output_names)
        const_nodes = GraphUtil._parse_graph_initializer(g, graph_proto)
        GraphUtil._parse_graph_input(
            g, graph_proto, [n.name for n in const_nodes])

        for n in g.get_nodes():
            for attr_name, attr_val in n.attr.items():
                if attr_val.HasField('g'):
                    # it was assumed that the a.g has inferred shapes/dtypes.
                    sub_g = GraphUtil.create_graph_from_onnx_graph(
                        attr_val.g, opset_version, extra_opset)
                    n.set_body_graph_as_attr(attr_name, sub_g)
        return g

    @staticmethod
    def get_node_count_from_onnx_graph(graph_proto):
        op_cnt = collections.Counter()
        for n in graph_proto.node:
            op_cnt[n.op_type] += 1
        return op_cnt

    @staticmethod
    def _parse_shape_and_type_from_value_infos(value_infos):
        """Get nodes output shapes and types from value infos."""
        output_shapes = {}
        output_dtypes = {}
        for shape_info in value_infos:
            type_proto = shape_info.type
            elem_type = type_proto.tensor_type.elem_type
            shape = type_proto.tensor_type.shape
            tuned_shape = []
            for d in shape.dim:
                if d.HasField('dim_param'):
                    tuned_shape.append(-1)
                elif d.HasField('dim_value'):
                    tuned_shape.append(d.dim_value)
                else:
                    # it is found, some unknown dims is missing after inference.
                    tuned_shape.append(-1)
            output_shapes[shape_info.name] = tuned_shape
            output_dtypes[shape_info.name] = elem_type

        return output_shapes, output_dtypes

    @staticmethod
    def _parse_graph_initializer(g, graph_proto):
        """Get graph initializers and put into Graph object."""
        const_nodes = []
        for initializer in graph_proto.initializer:
            np_val = numpy_helper.to_array(initializer)
            const_nodes.append(g.make_const(initializer.name, np_val))

        return const_nodes

    @staticmethod
    def _parse_graph_input(g, graph_proto, const_node_names):
        """Get graph inputs not defined as initializers and put into Graph object."""
        shapes, dtypes = GraphUtil._parse_shape_and_type_from_value_infos(
            graph_proto.input)
        # make sure the input is added in order we read from graph_proto,
        # because for subgraphs, the input orders matter.
        for graph_input in graph_proto.input:
            name = graph_input.name
            shape = shapes[name]
            dtype = dtypes[name]
            if name not in const_node_names:
                g.add_graph_input(name, dtype, shape)
            else:
                g.add_graph_input_with_default(
                    name, g.get_node_by_name(name), dtype, shape)
