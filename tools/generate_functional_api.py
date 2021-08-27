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
import os
import re
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--yaml_file_path",
    type=str,
    default="oneflow/core/functional/functional_api.yaml",
    help="The yaml format file that helps to generate the functional api or pybind cpp.",
)
parser.add_argument(
    "--generate_pybind",
    action="store_true",
    default=False,
    help="Enable to generate functional pybind cpp files.",
)
args = parser.parse_args()

api_generate_dir = "oneflow/core/functional"
pybind_generate_dir = "oneflow/api/python/functional"

license = """/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the \"License\");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an \"AS IS\" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Generated from oneflow/core/functional/functional_api.yaml. DO NOT EDIT!"""

header_fmt = (
    license
    + """

#ifndef ONEFLOW_CORE_FUNCTIONAL_GENERATED_FUNCTIONAL_API_H_
#define ONEFLOW_CORE_FUNCTIONAL_GENERATED_FUNCTIONAL_API_H_

#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/functional/tensor_index.h"

namespace oneflow {{
namespace one {{
namespace functional {{
{0}
}}  // namespace functional
}}  // namespace one
}}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_GENERATED_FUNCTIONAL_API_H_"""
)

source_fmt = (
    license
    + """

#include "{0}/functional_api.yaml.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {{
namespace one {{
namespace functional {{
{1}
}}  // namespace functional
}}  // namespace one
}}  // namespace oneflow
"""
)

pybind_fmt = (
    license
    + """

#include <vector>
#include <pybind11/pybind11.h>

#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/functional/function_def.h"
#include "oneflow/api/python/functional/py_function.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {{
namespace one {{
namespace functional {{
{0}
}}  // namespace functional
}}  // namespace one

namespace functional = one::functional;

ONEFLOW_API_PYBIND11_MODULE("F", m) {{
  py::options options;
  options.disable_function_signatures();

{1}
  options.enable_function_signatures();
}}

}}  // namespace oneflow
"""
)

types_allowed = {
    "Void",
    "Tensor",
    "TensorTuple",
    "Scalar",
    "Int",
    "Int32",
    "Int64",
    "Float",
    "Double",
    "String",
    "Bool",
    "ScalarList",
    "IntList",
    "Int32List",
    "Int64List",
    "FloatList",
    "DoubleList",
    "StringList",
    "BoolList",
    "DataType",
    "Shape",
    "Generator",
    "TensorIndex",
    "Device",
    "Placement",
    "Sbp",
    "SbpList",
}

generic_type_aliases = {
    "Int": "int32_t",
    "Int32": "int32_t",
    "Int64": "int64_t",
    "Float": "float",
    "Double": "double",
    "Bool": "bool",
}

argument_type_aliases = {
    "Tensor": "const std::shared_ptr<one::Tensor>&",
    "TensorTuple": "const TensorTuple&",
    "Scalar": "const Scalar&",
    "ScalarList": "const std::vector<Scalar>&",
    "IntList": "const std::vector<int32_t>&",
    "Int32List": "const std::vector<int32_t>&",
    "Int64List": "const std::vector<int64_t>&",
    "FloatList": "const std::vector<float>&",
    "DoubleList": "const std::vector<double>&",
    "String": "const std::string&",
    "StringList": "const std::vector<std::string>&",
    "BoolList": "const std::vector<bool>&",
    "DataType": "const Symbol<DType>&",
    "Shape": "const Shape&",
    "Generator": "const std::shared_ptr<one::Generator>&",
    "TensorIndex": "const TensorIndex&",
    "Device": "const Symbol<Device>&",
    "Placement": "const Symbol<ParallelDesc>&",
    "Sbp": "const Symbol<cfg::SbpParallel>&",
    "SbpList": "const std::vector<Symbol<cfg::SbpParallel>>&",
    **generic_type_aliases,
}

optional_argument_type_aliases = {
    "Tensor": "const Optional<one::Tensor>&",
    "TensorTuple": "const Optional<TensorTuple>&",
    "Scalar": "const Optional<Scalar>&",
    "ScalarList": "const Optional<std::vector<Scalar>>&",
    "IntList": "const Optional<std::vector<int32_t>>&",
    "Int32List": "const Optional<std::vector<int32_t>>&",
    "Int64List": "const Optional<std::vector<int64_t>>&",
    "FloatList": "const Optional<std::vector<float>>&",
    "DoubleList": "const Optional<std::vector<double>>&",
    "String": "const Optional<std::string>&",
    "StringList": "const Optional<std::vector<std::string>>&",
    "BoolList": "const Optional<std::vector<bool>>&",
    "DataType": "const Optional<Symbol<DType>>&",
    "Shape": "const Optional<Shape>&",
    "Generator": "const Optional<one::Generator>&",
    "TensorIndex": "const Optional<TensorIndex>&",
    "Device": "const Optional<Symbol<Device>>&",
    "Placement": "const Optional<Symbol<ParallelDesc>>&",
    "Sbp": "const Optional<Symbol<SbpParallel>>&",
    "SbpList": "const Optional<std::vector<Symbol<cfg::SbpParallel>>>&",
    **{k: "const Optional<{0}>".format(v) for k, v in generic_type_aliases.items()},
}

return_type_aliases = {
    "Void": "Maybe<void>",
    "Tensor": "Maybe<one::Tensor>",
    "TensorTuple": "Maybe<one::TensorTuple>",
    "String": "Maybe<std::string>",
    **{k: "Maybe<{0}>".format(v) for k, v in generic_type_aliases.items()},
}

value_aliases = {
    "True": "true",
    "False": "false",
    "kInt": "DType::Int32()",
    "kInt32": "DType::Int32()",
    "kInt64": "DType::Int64()",
    "kFloat": "DType::Float()",
    "kDouble": "DType::Double()",
    "kBool": "DType::Bool()",
}


def _escape_quote(fmt):
    return re.sub(r"\"|\'", '\\"', fmt)


def _normalize(fmt):
    fmt = fmt.strip()
    return re.sub(r"\s+", " ", fmt)


def _std_decay(fmt):
    fmt = fmt.strip()
    fmt = re.sub(r"(const|&)", "", fmt)
    return _normalize(fmt)


def parse_function_params(fmt):
    params = []
    fmt = _normalize(fmt)
    open_paren = fmt.find("(")
    if open_paren == -1:
        raise ValueError('Missing "(" in function def: ' + fmt)

    header = _normalize(fmt[0:open_paren])
    items = header.split(" ")
    if (len(items)) != 1:
        raise ValueError(
            "Missing return type or more than 1 return type in function def: " + fmt
        )
    params.append(items[0])

    close_paren = fmt.rfind(")")
    if close_paren == -1:
        raise ValueError('Missing ")" in Missingfunction def: ' + fmt)

    tail = fmt[open_paren + 1 : close_paren]
    # TODO(): Parse the parameter list more comprehensively.
    items = tail.split(",")
    for param in items:
        params.append(_normalize(param))

    pos = fmt.rfind("=>")
    if pos == -1:
        raise ValueError('Missing "=>" in Missingfunction def: ' + fmt)
    function_name = _normalize(fmt[pos + 2 :])
    return function_name, params


def render_file_if_different(target_file, content):
    if not os.path.isfile(target_file):
        with open(target_file, "w") as f:
            f.write(content)
    else:
        old_content = None
        with open(target_file, "r") as f:
            old_content = f.read()
        if old_content is None or old_content != content:
            with open(target_file, "w") as f:
                f.write(content)


class Argument:
    def __init__(self, fmt, keyword_only=False):
        self._keyword_only = keyword_only
        self._type = None
        self._name = None
        self._default_value = None

        fmt = _normalize(fmt)
        sp = fmt.rfind(" ")
        if sp == -1:
            raise ValueError("Missing argument type or name for argument def: " + fmt)
        self._type = _normalize(fmt[0:sp])
        assert self._type in types_allowed, "Unknow type: " + self._type

        self._optional = False
        self._name = _normalize(fmt[sp + 1 :])
        sp = self._name.find("=")
        if sp != -1:
            self._default_value = _normalize(self._name[sp + 1 :])
            if self._default_value == "None":
                self._optional = True
                self._default_cpp_value = ""
            elif self._default_value in value_aliases:
                self._default_cpp_value = value_aliases[self._default_value]
            else:
                self._default_cpp_value = self._default_value
            self._name = _normalize(self._name[0:sp])

        if not self._optional and self._type in argument_type_aliases:
            self._cpp_type = argument_type_aliases[self._type]
        elif self._optional and self._type in optional_argument_type_aliases:
            self._cpp_type = optional_argument_type_aliases[self._type]
        else:
            self._cpp_type = self._type

    @property
    def has_default_value(self):
        return self._default_value is not None

    def to_string(self, to_cpp=False):
        fmt = "{0} {1}".format(self._cpp_type if to_cpp else self._type, self._name)
        if not to_cpp and self.has_default_value:
            fmt += "={0}".format(self._default_value)
        return fmt


class Return:
    def __init__(self, fmt):
        self._type = _normalize(fmt)
        assert self._type in types_allowed, "Unknow type: " + self._type

        if self._type in return_type_aliases:
            self._cpp_type = return_type_aliases[self._type]
        else:
            self._cpp_type = self._type

    @property
    def type(self):
        return self._type

    def to_string(self, to_cpp=False):
        return self._cpp_type if to_cpp else self._type


class FunctionSignature:
    def __init__(self, fmt):
        self._fmt = fmt
        self._name, self._params = parse_function_params(fmt)
        self._ret = Return(self._params[0])
        keyword_only = False
        self._args = []
        self._max_positional_args_count = 0
        for arg in self._params[1:]:
            if arg == "*":
                keyword_only = True
                continue
            self._args.append(Argument(arg, keyword_only=keyword_only))
            if not keyword_only:
                self._max_positional_args_count += 1

        self._max_args_count = len(self._args)
        count = 0
        for arg in self._args:
            if arg._keyword_only:
                count += 1
        self._max_keyword_args_count = count

    @property
    def num_of_args():
        return len(self._args)

    def to_string(self, to_cpp=False, drop_name=False):
        if drop_name:
            fmt = "{0} (".format(self._ret.to_string(to_cpp=to_cpp))
        else:
            fmt = "{0} {1}(".format(self._ret.to_string(to_cpp=to_cpp), self._name)
        keyword_start = False
        for i, arg in enumerate(self._args):
            if i > 0 and i < len(self._args):
                fmt += ", "
            if not keyword_start and arg._keyword_only:
                keyword_start = True
                if not to_cpp:
                    fmt += "*, "
            fmt += arg.to_string(to_cpp=to_cpp)
        fmt += ")"
        return fmt


class Block:
    def __init__(self, name, signature, bind_python):
        self._name = name
        self._signature = signature
        self._bind_python = bind_python


class FunctionalGenerator:
    def __init__(self, input_file):
        self._blocks = {}
        with open(input_file) as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)
            for block in doc:
                assert "name" in block
                assert "signature" in block
                name = block["name"]
                signature = block["signature"]
                bind_python = False
                if "bind_python" in block:
                    bind_python = block["bind_python"]
                self._blocks[name] = list()
                if isinstance(signature, list):
                    for s in signature:
                        self._blocks[name].append(
                            Block(name, FunctionSignature(s), bind_python)
                        )
                else:
                    self._blocks[name].append(
                        Block(name, FunctionSignature(signature), bind_python)
                    )

    def generate_cpp_header_file(self, target_header_file):
        fmt = ""
        for name, blocks in self._blocks.items():
            for block in blocks:
                fmt += "\n"
                fmt += block._signature.to_string(to_cpp=True)
                fmt += ";\n"

        render_file_if_different(target_header_file, header_fmt.format(fmt))

    def generate_cpp_source_file(self, target_source_file):
        fmt = ""
        for name, blocks in self._blocks.items():
            for block in blocks:
                signature = block._signature
                fmt += "\n"
                fmt += signature.to_string(to_cpp=True)
                fmt += " {\n"
                fmt += '  static thread_local const auto& op = CHECK_JUST(FunctionLibrary::Global()->find("{0}"));\n'.format(
                    signature._name
                )
                fmt += "  return op->call<{0}, {1}>({2});\n".format(
                    signature._ret._cpp_type,
                    ", ".join([arg._cpp_type for arg in signature._args]),
                    ", ".join([arg._name for arg in signature._args]),
                )
                fmt += "}\n"

        render_file_if_different(
            target_source_file, source_fmt.format(api_generate_dir, fmt)
        )

    def generate_pybind_for_python(self, target_pybind_source_file):
        schema_fmt = ""
        module_fmt = ""
        for name, blocks in self._blocks.items():
            schema_types = []
            for block in blocks:
                if not block._bind_python:
                    continue
                signature = block._signature
                schema_types.append("functional::{0}Schema".format(signature._name))
                return_type = signature._ret._cpp_type
                schema_fmt += "\n"
                schema_fmt += "struct {0}Schema {{\n".format(signature._name)
                schema_fmt += "  using FType = decltype(functional::{0});\n".format(
                    signature._name
                )
                schema_fmt += "  using R = {0};\n".format(return_type)
                schema_fmt += "\n"
                schema_fmt += "  static constexpr FType* func = &functional::{0};\n".format(
                    signature._name
                )
                schema_fmt += "  static constexpr size_t max_args = {0};\n".format(
                    signature._max_args_count
                )
                schema_fmt += "  static constexpr size_t max_pos_args = {0};\n".format(
                    signature._max_positional_args_count
                )
                schema_fmt += '  static constexpr char const* signature = "{0}";\n'.format(
                    _escape_quote(signature.to_string(drop_name=True))
                )
                schema_fmt += "  static FunctionDef function_def;\n"
                schema_fmt += "};\n"
                schema_fmt += "\n"
                schema_fmt += "constexpr size_t {0}Schema::max_args;\n".format(
                    signature._name
                )
                schema_fmt += "constexpr size_t {0}Schema::max_pos_args;\n".format(
                    signature._name
                )
                schema_fmt += "constexpr char const* {0}Schema::signature;\n".format(
                    signature._name
                )
                return_def = "ReturnDef(ValueTypeOf<{0}>())".format(return_type)
                argument_def = []
                for arg in signature._args:
                    keyword_only = "true" if arg._keyword_only else "false"
                    optional = "true" if arg._optional else "false"
                    if arg.has_default_value:
                        argument_def.append(
                            '  ArgumentDef(/*name*/"{0}", /*default_value*/{1}({2}), /*keyword_only*/{3}, /*optional*/{4})'.format(
                                arg._name,
                                _std_decay(arg._cpp_type),
                                arg._default_cpp_value,
                                keyword_only,
                                optional,
                            )
                        )
                    else:
                        argument_def.append(
                            '  ArgumentDef(/*name*/"{0}", /*value_type*/ValueTypeOf<{1}>(), /*keyword_only*/{2})'.format(
                                arg._name, _std_decay(arg._cpp_type), keyword_only
                            )
                        )
                schema_fmt += 'FunctionDef {0}Schema::function_def = {{\n/*name*/"{1}",\n/*return_def*/{2},\n/*argument_def*/{{\n{3}\n}}\n}};\n'.format(
                    signature._name, name, return_def, ",\n".join(argument_def)
                )

            if len(schema_types) > 0:
                module_fmt += '  m.def("{0}", &functional::PyFunction<{1}>);\n'.format(
                    name, ", ".join(schema_types)
                )

        render_file_if_different(
            target_pybind_source_file, pybind_fmt.format(schema_fmt, module_fmt)
        )


if __name__ == "__main__":
    assert os.path.isfile(args.yaml_file_path), (
        "It is not a regular file for the yaml file which is " + args.yaml_file_path
    )
    g = FunctionalGenerator(args.yaml_file_path)

    assert os.path.isdir(api_generate_dir), (
        "Could not locate the api generate directory which is " + api_generate_dir
    )
    target_header_file = os.path.join(api_generate_dir, "functional_api.yaml.h")
    g.generate_cpp_header_file(target_header_file)
    target_source_file = os.path.join(api_generate_dir, "functional_api.yaml.cpp")
    g.generate_cpp_source_file(target_source_file)

    if args.generate_pybind:
        assert os.path.isdir(pybind_generate_dir), (
            "Could not locate the pybind generate directory which is "
            + pybind_generate_dir
        )
        target_pybind_source_file = os.path.join(
            pybind_generate_dir, "functional_api.yaml.pybind.cpp"
        )
        g.generate_pybind_for_python(target_pybind_source_file)
