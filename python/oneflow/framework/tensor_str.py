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
"""
This file is mostly referenced from PyTorch v1.8.1 torch/_tensor_str.py
"""


import math
import numpy as np
from typing import Optional
import oneflow as flow
from oneflow.framework.tensor_str_util import _autoset_linewidth
from oneflow.framework.tensor_str_util import _try_convert_to_local_tensor


class __PrinterOptions(object):
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    userset_linewidth: int = None
    sci_mode: Optional[bool] = None

    autoset_linewidth: bool = True

    @property
    def linewidth(self):
        return (
            _autoset_linewidth() if self.autoset_linewidth else self.userset_linewidth
        )

    @linewidth.setter
    def linewidth(self, value):
        self.userset_linewidth = value


PRINT_OPTS = __PrinterOptions()


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
):
    r"""Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = terminal_columns).
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is defined by
            `oneflow._tensor_str._Formatter`. This value is automatically chosen
            by the framework.
    .. note::
        linewidth equals to terminal columns, manual setting will invalidate the default automatic setting.
    """
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = math.inf
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80

    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode
    if profile is not None or linewidth is not None:
        PRINT_OPTS.autoset_linewidth = False


class _Formatter(object):
    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1
        self.random_sample_num = 50
        tensor = _try_convert_to_local_tensor(tensor)

        with flow.no_grad():
            tensor_view = tensor.reshape(-1)

        if not self.floating_dtype:
            for value in tensor_view:
                value_str = "{}".format(value)
                self.max_width = max(self.max_width, len(value_str))

        else:
            nonzero_finite_vals = flow.masked_select(tensor_view, tensor_view.ne(0))
            if nonzero_finite_vals.numel() == 0:
                # no valid number, do nothing
                return

            nonzero_finite_abs = nonzero_finite_vals.abs()
            nonzero_finite_min = nonzero_finite_abs.min().numpy().astype(np.float64)
            nonzero_finite_max = nonzero_finite_abs.max().numpy().astype(np.float64)

            for value in nonzero_finite_abs.numpy():
                if value != np.ceil(value):
                    self.int_mode = False
                    break

            if self.int_mode:
                # Check if scientific representation should be used.
                if (
                    nonzero_finite_max / nonzero_finite_min > 1000.0
                    or nonzero_finite_max > 1.0e8
                ):
                    self.sci_mode = True
                    for value in nonzero_finite_vals:
                        value_str = (
                            ("{{:.{}e}}").format(PRINT_OPTS.precision).format(value)
                        )
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = ("{:.0f}").format(value)
                        self.max_width = max(self.max_width, len(value_str) + 1)
            else:
                if (
                    nonzero_finite_max / nonzero_finite_min > 1000.0
                    or nonzero_finite_max > 1.0e8
                    or nonzero_finite_min < 1.0e-4
                ):
                    self.sci_mode = True
                    for value in nonzero_finite_vals:
                        value_str = (
                            ("{{:.{}e}}").format(PRINT_OPTS.precision).format(value)
                        )
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = (
                            ("{{:.{}f}}").format(PRINT_OPTS.precision).format(value)
                        )
                        self.max_width = max(self.max_width, len(value_str))

        if PRINT_OPTS.sci_mode is not None:
            self.sci_mode = PRINT_OPTS.sci_mode

    def width(self):
        return self.max_width

    def format(self, value):
        if self.floating_dtype:
            if self.sci_mode:
                ret = (
                    ("{{:{}.{}e}}")
                    .format(self.max_width, PRINT_OPTS.precision)
                    .format(value)
                )
            elif self.int_mode:
                ret = "{:.0f}".format(value)
                if not (math.isinf(value) or math.isnan(value)):
                    ret += "."
            else:
                ret = ("{{:.{}f}}").format(PRINT_OPTS.precision).format(value)
        else:
            ret = "{}".format(value)
        return (self.max_width - len(ret)) * " " + ret


def _scalar_str(self, formatter1):
    return formatter1.format(_try_convert_to_local_tensor(self).tolist())


def _vector_str(self, indent, summarize, formatter1):
    # length includes spaces and comma between elements
    element_length = formatter1.width() + 2
    elements_per_line = max(
        1, int(math.floor((PRINT_OPTS.linewidth - indent) / (element_length)))
    )

    def _val_formatter(val, formatter1=formatter1):
        return formatter1.format(val)

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        left_values = _try_convert_to_local_tensor(
            self[: PRINT_OPTS.edgeitems]
        ).tolist()
        right_values = _try_convert_to_local_tensor(
            self[-PRINT_OPTS.edgeitems :]
        ).tolist()
        data = (
            [_val_formatter(val) for val in left_values]
            + [" ..."]
            + [_val_formatter(val) for val in right_values]
        )
    else:
        values = _try_convert_to_local_tensor(self).tolist()
        data = [_val_formatter(val) for val in values]

    data_lines = [
        data[i : i + elements_per_line] for i in range(0, len(data), elements_per_line)
    ]
    lines = [", ".join(line) for line in data_lines]
    return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"


def _tensor_str_with_formatter(self, indent, summarize, formatter1):
    dim = self.dim()

    if dim == 0:
        return _scalar_str(self, formatter1)

    if dim == 1:
        return _vector_str(self, indent, summarize, formatter1)

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = (
            [
                _tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1,)
                for i in range(0, PRINT_OPTS.edgeitems)
            ]
            + ["..."]
            + [
                _tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1,)
                for i in range(self.shape[0] - PRINT_OPTS.edgeitems, self.shape[0])
            ]
        )
    else:
        slices = [
            _tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1)
            for i in range(0, self.size(0))
        ]

    tensor_str = ("," + "\n" * (dim - 1) + " " * (indent + 1)).join(slices)
    return "[" + tensor_str + "]"


def _tensor_str(self, indent):
    summarize = self.numel() > PRINT_OPTS.threshold
    if self.dtype is flow.float16:
        self = self.float()

    with flow.no_grad():
        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        return _tensor_str_with_formatter(self, indent, summarize, formatter)


def _add_suffixes(tensor_str, suffixes, indent):
    tensor_strs = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind("\n") + 1
    for suffix in suffixes:
        suffix_len = len(suffix)
        if last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            tensor_strs.append(",\n" + " " * indent + suffix)
            last_line_len = indent + suffix_len
        else:
            tensor_strs.append(", " + suffix)
            last_line_len += suffix_len + 2
    tensor_strs.append(")")
    return "".join(tensor_strs)


def get_summarized_data(self):
    dim = self.dim()
    if dim == 0:
        return self
    if dim == 1:
        if self.size(0) > 2 * PRINT_OPTS.edgeitems:
            return flow.cat(
                (self[: PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems :])
            )
        else:
            return self
    if self.size(0) > 2 * PRINT_OPTS.edgeitems:
        start = [self[i] for i in range(0, PRINT_OPTS.edgeitems)]
        end = [
            self[i] for i in range(self.shape[0] - PRINT_OPTS.edgeitems, self.shape[0])
        ]
        return flow.stack([get_summarized_data(x) for x in (start + end)])
    else:
        return flow.stack([get_summarized_data(x) for x in self])


def _format_tensor_on_cpu(tensor):
    if tensor.is_global:
        device = tensor.placement.type
    else:
        device = tensor.device.type
    return device != "cpu" and device != "cuda"


def _gen_tensor_str_template(tensor, is_meta):
    is_meta = is_meta or tensor.is_lazy
    prefix = "tensor("
    indent = len(prefix)
    suffixes = []

    # tensor is local or global
    if tensor.is_global:
        suffixes.append(f"placement={str(tensor.placement)}")
        suffixes.append(f"sbp={str(tensor.sbp)}")
    elif tensor.device.type != "cpu":
        suffixes.append("device='" + str(tensor.device) + "'")
    if tensor.is_lazy:
        suffixes.append("is_lazy='True'")

    # tensor is empty, meta or normal
    if tensor.numel() == 0:
        # Explicitly print the shape if it is not (0,), to match NumPy behavior
        if tensor.dim() != 1:
            suffixes.append("size=" + str(tuple(tensor.shape)))
        tensor_str = "[]"
    elif is_meta:
        tensor_str = "..."
        suffixes.append("size=" + str(tuple(tensor.shape)))
    else:
        if _format_tensor_on_cpu(tensor):
            tensor_str = _tensor_str(tensor.detach().to("cpu"), indent)
        else:
            tensor_str = _tensor_str(tensor, indent)

    suffixes.append("dtype=" + str(tensor.dtype))
    if tensor.grad_fn is not None:
        name = tensor.grad_fn.name()
        suffixes.append("grad_fn=<{}>".format(name))
    elif tensor.requires_grad:
        suffixes.append("requires_grad=True")

    return _add_suffixes(prefix + tensor_str, suffixes, indent)


def _gen_tensor_str(tensor):
    return _gen_tensor_str_template(tensor, False)


def _gen_tensor_meta_str(tensor):
    # meta
    return _gen_tensor_str_template(tensor, True)
