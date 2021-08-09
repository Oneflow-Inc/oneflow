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
import numpy as np
import oneflow as flow

import math

from typing import Optional


class __PrinterOptions(object):
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: Optional[bool] = None


PRINT_OPTS = __PrinterOptions()


def _convert_to_local_tensor(self):
    # consistent to local
    if self.is_consistent:
        placement = flow.placement("cpu", {0: [0]})
        sbp = flow.sbp.broadcast
        # TODO: delete `to("cuda")` after supporting cpu data broadcast
        self = self.to("cuda").to_consistent(placement, sbp).to_local()
    return self


class _Formatter(object):
    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1
        self.random_sample_num = 50

        with flow.no_grad():
            tensor_view = tensor.reshape([-1])

        if not self.floating_dtype:
            # TODO: verify int type tensor
            max_value = tensor.abs().max().numpy()
            value_str = "{}".format(max_value)
            self.max_width = max(self.max_width, len(value_str))

        else:
            nonzero_finite_vals = flow.masked_select(tensor_view, tensor_view.ne(0))
            if tensor_view.numel() == 0:
                # no valid number, do nothing
                return

            nonzero_finite_abs = nonzero_finite_vals.abs()
            nonzero_finite_min = nonzero_finite_abs.min().numpy()
            nonzero_finite_max = nonzero_finite_abs.max().numpy()

            # Determines use int mode or not
            self.random_sample_num = min(self.random_sample_num, tensor_view.numel())
            rand_idx = np.random.randint(
                tensor_view.numel(), size=[self.random_sample_num]
            ).tolist()
            sample_data = nonzero_finite_abs[rand_idx].numpy()
            for value in sample_data:
                if value != np.ceil(value):
                    self.int_mode = False
                    break

            if self.int_mode:
                # in int_mode for floats, all numbers are integers, and we append a decimal to nonfinites
                # to indicate that the tensor is of floating type. add 1 to the len to account for this.
                if (
                    nonzero_finite_max / nonzero_finite_min > 1000.0
                    or nonzero_finite_max > 1.0e8
                ):
                    self.sci_mode = True
                    value_str = (
                        ("{{:.{}e}}")
                        .format(PRINT_OPTS.precision)
                        .format(nonzero_finite_max)
                    )
                    self.max_width = max(self.max_width, len(value_str))
                else:
                    value_str = ("{:.0f}").format(nonzero_finite_max)
                    self.max_width = max(self.max_width, len(value_str))
            else:
                # Check if scientific representation should be used.
                if (
                    nonzero_finite_max / nonzero_finite_min > 1000.0
                    or nonzero_finite_max > 1.0e8
                    or nonzero_finite_min < 1.0e-4
                ):
                    self.sci_mode = True
                    for value in (nonzero_finite_max, nonzero_finite_min):
                        value_str = (
                            ("{{:.{}e}}").format(PRINT_OPTS.precision).format(value)
                        )
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    value_str = (
                        ("{{:.{}f}}")
                        .format(PRINT_OPTS.precision)
                        .format(nonzero_finite_max)
                    )
                    self.max_width = max(self.max_width, len(value_str))

        # add singal position
        self.max_width += 1

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
    return formatter1.format(self.tolist())


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
        data = (
            [_val_formatter(val) for val in self[: PRINT_OPTS.edgeitems].tolist()]
            + [" ..."]
            + [_val_formatter(val) for val in self[-PRINT_OPTS.edgeitems :].tolist()]
        )
    else:
        data = [_val_formatter(val) for val in self.tolist()]

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

    # TODO: convert to local before slicing consistent tensor
    # self = _convert_to_local_tensor(self)
    if self.is_consistent:
        return "[]"

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


def cat_data(inp):
    return flow.cat((inp[: PRINT_OPTS.edgeitems], inp[-PRINT_OPTS.edgeitems :]))


def get_summarized_data(self):
    # TODO: supports consistent slice and delete this assert
    assert self.is_local

    dim = self.dim()
    if dim == 0:
        return self
    if dim == 1:
        if self.size(0) > 2 * PRINT_OPTS.edgeitems:
            return cat_data(self)
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


def _gen_tensor_str(inp):
    prefix = "tensor("
    indent = len(prefix)
    suffixes = []
    if inp.is_consistent:
        suffixes.append(f"placement={str(inp.placement)}")
        suffixes.append(f"sbp={str(inp.sbp)}")
    elif inp.device.type != "cpu" or (
        inp.device.type == "cuda" and inp.device.index != 0
    ):
        suffixes.append("device='" + str(inp.device) + "'")
    if inp.numel() == 0:
        # Explicitly print the shape if it is not (0,), to match NumPy behavior
        if inp.dim() != 1:
            suffixes.append("size=" + str(tuple(inp.shape)))
        suffixes.append("dtype=" + str(inp.dtype))
        tensor_str = "[]"
    else:
        suffixes.append("dtype=" + str(inp.dtype))
        tensor_str = _tensor_str(inp, indent)
    if inp.grad_fn is not None:
        name = tensor.grad_fn.name()
        suffixes.append("grad_fn=<{}>".format(name))
    elif inp.requires_grad:
        suffixes.append("requires_grad=True")

    return _add_suffixes(prefix + tensor_str, suffixes, indent)
