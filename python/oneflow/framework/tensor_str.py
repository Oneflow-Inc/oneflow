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
import numpy as np
import math
from typing import Optional

import oneflow as flow


class __PrinterOptions(object):
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: Optional[bool] = None


PRINT_OPTS = __PrinterOptions()


def _try_convert_to_local_tensor(tensor):
    if tensor.is_consistent:
        tensor = tensor.to_consistent(
            placement=tensor.placement, sbp=flow.sbp.broadcast
        ).to_local()
    return tensor


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
    char_per_line = element_length * elements_per_line

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
                _tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1)
                for i in range(0, PRINT_OPTS.edgeitems)
            ]
            + ["..."]
            + [
                _tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1)
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

    # TODO: not support flow.sbp.split(x) but flow.sbp.split(0).
    def _cannot_print(sbp):
        return (
            sbp != flow.sbp.partial_sum
            and sbp != flow.sbp.broadcast
            and sbp != flow.sbp.split(0)
        )

    # TODO: delete it when boxing on "CPU" and s1->b on "GPU" are ready
    if self.is_consistent:
        self = self.to("cuda")
        if all(_cannot_print(sbp) for sbp in self.sbp):
            return "[...]"

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


def _gen_tensor_str_template(tensor, is_meta):
    is_meta = is_meta or tensor.is_lazy
    prefix = "tensor("
    indent = len(prefix)
    suffixes = []

    # tensor is local or consistent
    if tensor.is_consistent:
        suffixes.append(f"placement={str(tensor.placement)}")
        suffixes.append(f"sbp={str(tensor.sbp)}")
    elif tensor.device.type == "cuda" or tensor.device.type == "gpu":
        suffixes.append("device='" + str(tensor.device) + "'")
    elif tensor.device.type != "cpu":
        raise RunTimeError("unknow device type")
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
