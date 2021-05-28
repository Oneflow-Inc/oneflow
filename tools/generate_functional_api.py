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
import yaml

api_generate_dir = "oneflow/core/functional"
schema_generate_dir = "oneflow/api/python/functional"

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
*/"""

header_fmt = (
    license
    + """

#ifndef ONEFLOW_CORE_FUNCTIONAL_GENERATED_FUNCTIONAL_API_H_
#define ONEFLOW_CORE_FUNCTIONAL_GENERATED_FUNCTIONAL_API_H_

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/scalar.h"

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

#include "oneflow/core/functional/generated/functional_api.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {{
namespace one {{
namespace functional {{
{0}
}}  // namespace functional
}}  // namespace one
}}  // namespace oneflow
"""
)

schema_source_fmt = (
    license
    + """
#include <vector>
#include <pybind11/pybind11.h>

#include "oneflow/api/python/of_api_registry.h"

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/functional/generated/functional_api.h"
#include "oneflow/api/python/functional/function_def.h"
#include "oneflow/api/python/functional/py_function.h"

namespace oneflow {{
namespace one {{
namespace functional {{
{0}
}}  // namespace functional
}}  // namespace one

namespace functional = one::functional;

ONEFLOW_API_PYBIND11_MODULE("F", m) {{
{1}
}}

}}  // namespace oneflow
"""
)

types_allowed = {
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
    **generic_type_aliases,
}

return_type_aliases = {
    "Tensor": "Maybe<one::Tensor>",
    "TensorTuple": "Maybe<one::TensorTuple>",
    "String": "std::string",
    **generic_type_aliases,
}

value_aliases = {
    "True": "true",
    "False": "false",
}


def normalize(fmt):
    fmt = fmt.strip()
    import re

    return re.sub(r"\s+", " ", fmt)


def std_decay(fmt):
    fmt = fmt.strip()
    import re

    fmt = re.sub(r"(const|&)", "", fmt)
    return normalize(fmt)


def parse_function_params(fmt):
    params = []
    fmt = normalize(fmt)
    if not fmt.endswith(")"):
        raise ValueError('Function def should end with ")": ' + fmt)
    open_paren = fmt.find("(")
    if open_paren == -1:
        raise ValueError('Missing "(" in function def: ' + fmt)

    header = fmt[0:open_paren]
    items = header.split(" ")
    if (len(items)) != 2:
        raise ValueError("Missing return type or function name in function def: " + fmt)
    params.append(items[0])
    function_name = items[1]

    tail = fmt[open_paren + 1 : -1]
    # TODO(): Parse the parameter list more comprehensively.
    items = tail.split(",")
    for param in items:
        params.append(normalize(param))
    return function_name, params


class Argument:
    def __init__(self, fmt, keyword_allowed=False):
        self._keyword_allowed = keyword_allowed
        self._type = None
        self._name = None
        self._default_value = None

        fmt = normalize(fmt)
        sp = fmt.find(" ")
        if sp == -1:
            raise ValueError("Missing argument type or name for argument def: " + fmt)
        self._type = normalize(fmt[0:sp])
        assert self._type in types_allowed, "Unknow type: " + self._type

        if self._type in argument_type_aliases:
            self._cpp_type = argument_type_aliases[self._type]
        else:
            self._cpp_type = self._type
        self._name = normalize(fmt[sp + 1 :])
        sp = self._name.find("=")
        if sp != -1:
            self._default_value = normalize(self._name[sp + 1 :])
            if self._default_value in value_aliases:
                self._default_cpp_value = value_aliases[self._default_value]
            else:
                self._default_cpp_value = self._default_value
            self._name = normalize(self._name[0:sp])

    @property
    def has_default_value(self):
        return self._default_value is not None

    def to_string(self, to_cpp=False):
        fmt = "{0} {1}".format(self._cpp_type if to_cpp else self.type, self._name)
        if not to_cpp and self.has_default_value:
            fmt += "={0}".format(self._default_value)
        return fmt


class Return:
    def __init__(self, fmt):
        self._type = normalize(fmt)
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
        keyword_allowed = False
        self._args = []
        for arg in self._params[1:]:
            if arg == "*":
                keyword_allowed = True
                continue
            self._args.append(Argument(arg, keyword_allowed=keyword_allowed))

        self._max_args_count = len(self._args)
        self._max_positional_args_count = self._max_args_count
        count = 0
        for arg in self._args:
            if arg._keyword_allowed:
                count += 1
        self._max_keyword_args_count = count

    @property
    def num_of_args():
        return len(self._args)

    def to_string(self, to_cpp=False):
        fmt = "{0} {1}(".format(self._ret.to_string(to_cpp=to_cpp), self._name)
        fmt += ", ".join([arg.to_string(to_cpp=to_cpp) for arg in self._args])
        fmt += ")"
        return fmt


class Block:
    def __init__(self, name, signature, bind_python):
        self._name = name
        self._signature = signature
        self._bind_python = bind_python


class FunctionGenerator:
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
                self._blocks[name] = Block(
                    name, FunctionSignature(signature), bind_python
                )

    def generate_cpp_header_file(self, target_header_file):
        fmt = ""
        for name, block in self._blocks.items():
            fmt += "\n"
            fmt += block._signature.to_string(to_cpp=True)
            fmt += ";\n"

        with open(target_header_file, "w") as f:
            f.write(header_fmt.format(fmt))

    def generate_cpp_source_file(self, target_source_file):
        fmt = ""
        for name, block in self._blocks.items():
            signature = block._signature
            fmt += "\n"
            fmt += signature.to_string(to_cpp=True)
            fmt += " {\n"
            fmt += '  static thread_local const auto& op = CHECK_JUST(FunctionLibrary::Global()->find("{0}"));\n'.format(
                signature._name
            )
            fmt += "  return op->call<{0}>({1});\n".format(
                signature._ret._cpp_type,
                ", ".join([arg._name for arg in signature._args]),
            )
            fmt += "}\n"

        with open(target_source_file, "w") as f:
            f.write(source_fmt.format(fmt))

    def generate_function_schema_for_python(self, target_schema_source_file):
        schema_fmt = ""
        module_fmt = ""
        for name, block in self._blocks.items():
            if not block._bind_python:
                continue

            signature = block._signature
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
            schema_fmt += "  static constexpr size_t max_positionals = {0};\n".format(
                signature._max_positional_args_count
            )
            schema_fmt += "  static constexpr size_t max_keywords = {0};\n".format(
                signature._max_keyword_args_count
            )
            schema_fmt += "  static ReturnDef return_def;\n"
            schema_fmt += "  static std::vector<ArgumentDef> argument_def;\n"
            schema_fmt += "};\n"
            schema_fmt += "\n"
            schema_fmt += "ReturnDef {0}Schema::return_def = ReturnDef(ValueTypeOf<{1}>());\n".format(
                signature._name, return_type,
            )

            argument_def = []
            for arg in signature._args:
                if arg.has_default_value:
                    argument_def.append(
                        'ArgumentDef("{0}", {1}({2}))'.format(
                            arg._name, std_decay(arg._cpp_type), arg._default_cpp_value
                        )
                    )
                else:
                    argument_def.append(
                        'ArgumentDef("{0}", ValueTypeOf<{1}>())'.format(
                            arg._name, std_decay(arg._cpp_type)
                        )
                    )

            schema_fmt += "std::vector<ArgumentDef> {0}Schema::argument_def = {{{1}}};\n".format(
                signature._name, ", ".join(argument_def)
            )
            module_fmt += '  m.def("{0}", &functional::PyFunction<functional::{1}Schema>);\n'.format(
                name, signature._name
            )

        with open(target_schema_source_file, "w") as f:
            f.write(schema_source_fmt.format(schema_fmt, module_fmt))


if __name__ == "__main__":
    yaml_file = os.path.join(api_generate_dir, "functional_api.yaml")
    assert os.path.isdir(api_generate_dir), (
        "Could not locate the api generate directory which path is " + api_generate_dir
    )
    assert os.path.isfile(yaml_file), (
        "It is not a regular file for the yaml file which path is " + yaml_file
    )
    g = FunctionGenerator(yaml_file)

    # Create api generated directory if needed.
    directory = os.path.join(api_generate_dir, "generated")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate functional api header and source file.
    target_header_file = os.path.join(directory, "functional_api.h")
    g.generate_cpp_header_file(target_header_file)
    target_source_file = os.path.join(directory, "functional_api.cpp")
    g.generate_cpp_source_file(target_source_file)

    # Create schema generated directory if needed.
    directory = os.path.join(schema_generate_dir, "generated")
    if not os.path.exists(directory):
        os.makedirs(directory)
    target_schema_source_file = os.path.join(directory, "functional_schema.cpp")
    g.generate_function_schema_for_python(target_schema_source_file)
