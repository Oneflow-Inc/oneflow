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
    "OpExpr",
    "PyObject*",
    "ShapeList",
    "DataTypeList",
    "Layout",
    "MemoryFormat",
}

mangled_name = {
    "Void": "V",
    "Tensor": "T",
    "TensorTuple": "Tt",
    "Scalar": "Sc",
    "Int": "I",
    "Int32": "I32",
    "Int64": "I64",
    "Float": "F",
    "Double": "D",
    "String": "S",
    "Bool": "B",
    "ScalarList": "Scl",
    "IntList": "Il",
    "Int32List": "I32l",
    "Int64List": "I64l",
    "FloatList": "Fl",
    "DoubleList": "Dl",
    "StringList": "Sl",
    "BoolList": "Bl",
    "DataType": "Dt",
    "Shape": "Sh",
    "Generator": "G",
    "TensorIndex": "Ti",
    "Device": "De",
    "Placement": "P",
    "Sbp": "Sbp",
    "SbpList": "Sbpl",
    "OpExpr": "Op",
    "PyObject*": "Pyo",
    "ShapeList": "Shl",
    "DataTypeList": "Dtl",
    "Layout": "Lo",
    "MemoryFormat": "Mf",
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
    "Sbp": "const Symbol<SbpParallel>&",
    "SbpList": "const std::vector<Symbol<SbpParallel>>&",
    "OpExpr": "const std::shared_ptr<one::OpExpr>&",
    "PyObject*": "PyObject*",
    "ShapeList": "const std::vector<Shape>&",
    "DataTypeList": "const std::vector<Symbol<DType>>&",
    "Layout": "const Symbol<Layout>&",
    "MemoryFormat": "const Symbol<MemoryFormat>&",
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
    "SbpList": "const Optional<std::vector<Symbol<SbpParallel>>>&",
    "OpExpr": "const Optional<one::OpExpr>&",
    "PyObject*": "const Optional<PyObject*>&",
    "ShapeList": "const Optional<std::vector<Shape>>&",
    "DataTypeList": "const Optional<std::vector<Symbol<DType>>>&",
    "Layout": "const Optional<Symbol<Layout>>&",
    "MemoryFormat": "const Optional<Symbol<MemoryFormat>>&",
    **{k: "const Optional<{0}>&".format(v) for k, v in generic_type_aliases.items()},
}

return_type_aliases = {
    "Void": "Maybe<void>",
    "Tensor": "Maybe<one::Tensor>",
    "TensorTuple": "Maybe<one::TensorTuple>",
    "String": "Maybe<std::string>",
    "Shape": "Maybe<Shape>",
    **{k: "Maybe<{0}>".format(v) for k, v in generic_type_aliases.items()},
}

value_aliases = {
    "True": "true",
    "False": "false",
    "kInt": "DType::Int32()",
    "kInt8": "DType::Int8()",
    "kUInt8": "DType::UInt8()",
    "kInt32": "DType::Int32()",
    "kInt64": "DType::Int64()",
    "kFloat": "DType::Float()",
    "kDouble": "DType::Double()",
    "kBool": "DType::Bool()",
    "kStrided": "Layout::Strided()",
    "kPreserve": "MemoryFormat::Preserve()",
    "kContiguous": "MemoryFormat::Contiguous()",
}


def _escape_quote(fmt):
    return re.sub(r"\"|\'", '\\"', fmt)


def _normalize(fmt):
    fmt = fmt.strip()
    return re.sub(r"\s+", " ", fmt)


def _remove_square_brackets_and_content_inside(fmt):
    # "TensorTuple[values], TensorTuple[indices]" -> "TensorTuple, TensorTuple"
    return re.sub(r"\[[^()]*?\]", "", fmt)


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
    items = _normalize(_remove_square_brackets_and_content_inside(header)).split(" ")
    if (len(items)) != 1:
        raise ValueError(
            "Missing return type or more than 1 return type in function def: " + fmt
        )

    params.append(header)

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


def generate_return_types_named_tuple(return_names, func_name, block_name):
    param_names = ", ".join(
        [
            '{{const_cast<char*>("{}"), const_cast<char*>("")}}'.format(x)
            for x in return_names
        ]
    )
    code = f"""PyTypeObject* Get{func_name}NamedTuple() {{
  static PyStructSequence_Field NamedTuple_fields[] = {{ {param_names},  {{nullptr}} }};
  static PyTypeObject {func_name}NamedTuple;
  static bool is_initialized = false;
  static PyStructSequence_Desc desc = {{ const_cast<char*>("oneflow.return_types.{block_name}"), nullptr, NamedTuple_fields, {len(return_names)} }};
  if (!is_initialized) {{
      PyStructSequence_InitType(&{func_name}NamedTuple, &desc);
      {func_name}NamedTuple.tp_repr = (reprfunc)returned_structseq_repr;
      is_initialized = true;
  }}
  return &{func_name}NamedTuple;
}}
"""
    return code


class Argument:
    def __init__(self, fmt, keyword_only=False):
        self._keyword_only = keyword_only
        self._type = None
        self._name = None
        self._default_value = None
        self._size = 0

        fmt = _normalize(fmt)
        sp = fmt.rfind(" ")
        if sp == -1:
            raise ValueError("Missing argument type or name for argument def: " + fmt)
        type_name = fmt[0:sp]
        arg_name = fmt[sp + 1 :]
        sp = type_name.find("[")
        if sp != -1:
            self._type = _normalize(type_name[0:sp])
            size = type_name[sp + 1 :]
            sp = size.find("]")
            assert sp != -1, "Missing ']' for argument def: " + fmt
            size = _normalize(size[0:sp])
            assert size.isnumeric(), (
                "list size is not an integer for argument def: " + fmt
            )
            self._size = int(size)
        else:
            self._type = _normalize(type_name)
        assert self._type in types_allowed, "Unknow type: " + self._type

        self._optional = False
        self._name = _normalize(arg_name)
        sp = self._name.find("=")
        if sp != -1:
            self._default_value = _normalize(self._name[sp + 1 :])
            if self._default_value == "None":
                self._optional = True
                self._default_cpp_value = ""
            elif self._type.endswith("List"):
                if self._default_value != "None":
                    _value_list = [
                        self._default_value for i in range(self._size)
                    ]  # For int32List[2] = 2, _value_list will be ["2", "2"]
                    self._default_cpp_value = (
                        "{" + ", ".join(_value_list) + "}"
                    )  # ["2", "2"] -> "{2, 2}"
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
        self._type, self._return_names = self.check_named_tuple(_normalize(fmt))
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

    def check_named_tuple(self, fmt):
        matches = re.match(r"(.*?)\s*\[(.*?)\]", fmt)
        if matches is None:
            type, return_names = _normalize(fmt), None
        else:
            type = matches.group(1)
            return_names = [_normalize(x) for x in matches.group(2).split(",")]
        return type, return_names


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

    def get_mangled_type(self):
        fmt = mangled_name[self._ret._type]
        for _, arg in enumerate(self._args):
            fmt += mangled_name[arg._type]
        return fmt

    def get_schema_name(self):
        return "{0}Schema_{1}".format(self._name, self.get_mangled_type())


class Block:
    def __init__(self, name, signature, bind_python):
        self._name = name
        self._signature = signature
        self._bind_python = bind_python


class Generator:
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

    def generate_cpp_header_file(self, header_fmt, target_header_file):
        fmt = ""
        for name, blocks in self._blocks.items():
            for block in blocks:
                fmt += "\n"
                fmt += block._signature.to_string(to_cpp=True)
                fmt += ";\n"

        render_file_if_different(target_header_file, header_fmt.format(fmt))

    def generate_cpp_source_file(self, source_fmt, target_source_file):
        fmt = ""
        for name, blocks in self._blocks.items():
            for block in blocks:
                signature = block._signature
                fmt += "\n"
                fmt += signature.to_string(to_cpp=True)
                fmt += " {\n"
                fmt += '  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<{0}, {1}>("{2}"));\n'.format(
                    signature._ret._cpp_type,
                    ", ".join([arg._cpp_type for arg in signature._args]),
                    signature._name,
                )
                fmt += "  return __op->call({0});\n".format(
                    ", ".join([arg._name for arg in signature._args]),
                )
                fmt += "}\n"

        render_file_if_different(target_source_file, source_fmt.format(fmt))

    def generate_pybind_for_python(
        self,
        pybind_header_fmt,
        pybind_source_fmt,
        target_pybind_header_file,
        target_pybind_source_file,
    ):
        schema_fmt = ""
        module_fmt = ""
        header_fmt = ""

        return_type_fmt = ""
        map_pairs = []
        for name, blocks in self._blocks.items():
            schema_types = []
            max_args_count = 0
            for block in blocks:
                if not block._bind_python:
                    continue
                signature = block._signature
                max_args_count = max(max_args_count, signature._max_args_count)
                schema_types.append(
                    "functional::{0}".format(signature.get_schema_name())
                )
                return_type = signature._ret._cpp_type
                schema_fmt += "\n"
                schema_fmt += "struct {0} {{\n".format(signature.get_schema_name())
                schema_fmt += "  using FType = {0};\n".format(
                    signature.to_string(to_cpp=True, drop_name=True)
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
                schema_fmt += "constexpr size_t {0}::max_args;\n".format(
                    signature.get_schema_name()
                )
                schema_fmt += "constexpr size_t {0}::max_pos_args;\n".format(
                    signature.get_schema_name()
                )
                schema_fmt += "constexpr char const* {0}::signature;\n".format(
                    signature.get_schema_name()
                )
                return_def = "ReturnDef(ValueTypeOf<{0}>())".format(return_type)
                argument_def = []
                for arg in signature._args:
                    keyword_only = "true" if arg._keyword_only else "false"
                    optional = "true" if arg._optional else "false"
                    if arg.has_default_value:
                        argument_def.append(
                            '  ArgumentDef(/*name*/"{0}", /*default_value*/{1}({2}), /*size*/{3}, /*keyword_only*/{4}, /*optional*/{5})'.format(
                                arg._name,
                                _std_decay(arg._cpp_type),
                                arg._default_cpp_value,
                                arg._size,
                                keyword_only,
                                optional,
                            )
                        )
                    else:
                        argument_def.append(
                            '  ArgumentDef(/*name*/"{0}", /*value_type*/ValueTypeOf<{1}>(), /*size*/{2}, /*keyword_only*/{3}, /*optional*/{4})'.format(
                                arg._name,
                                _std_decay(arg._cpp_type),
                                arg._size,
                                keyword_only,
                                optional,
                            )
                        )
                schema_fmt += 'FunctionDef {0}::function_def = {{\n/*name*/"{1}",\n/*return_def*/{2},\n/*argument_def*/{{\n{3}\n}}\n}};\n'.format(
                    signature.get_schema_name(),
                    name,
                    return_def,
                    ",\n".join(argument_def),
                )

            if len(schema_types) > 0:
                module_fmt += '    {{"{0}", (PyCFunction)functional::{1}, METH_VARARGS | METH_KEYWORDS, NULL}},\n'.format(
                    name, name
                )

                header_fmt += "\n"
                header_fmt += "PyObject* {0}(PyObject* self, PyObject* args, PyObject* kwargs);\n".format(
                    name
                )
                schema_fmt += "\n"
                schema_fmt += "PyObject* {0}(PyObject* self, PyObject* args, PyObject* kwargs) {{\n".format(
                    name
                )
                schema_fmt += "  HANDLE_ERRORS\n"
                schema_fmt += '  OF_PROFILER_RANGE_GUARD("{0}");\n'.format(name)
                schema_fmt += "  PythonFrameGuard pf;\n"
                schema_fmt += '  static PythonArgParser<{0}> parser("{1}");\n'.format(
                    ", ".join(schema_types), name
                )
                schema_fmt += "  ParsedArgs<{0}> r;\n".format(max_args_count)
                schema_fmt += "  int idx = parser.Parse(args, kwargs, &r);\n"
                i = 0
                for block in blocks:
                    signature = block._signature
                    schema_fmt += "  if (idx == {0}) {{\n".format(i)
                    params = []
                    for j in range(len(signature._args)):
                        cpp_type = _std_decay(signature._args[j]._cpp_type)
                        params.append("r[{0}].As<{1}>()".format(j, cpp_type))
                    if signature._ret._return_names is None:
                        schema_fmt += "    return CastToPyObject(functional::{0}({1}));\n".format(
                            signature._name, ", ".join(params)
                        )
                    else:
                        schema_fmt += '    return WrapTensorTuple(functional::{0}({1}).GetOrThrow(), "{2}");\n'.format(
                            signature._name, ", ".join(params), signature._name,
                        )
                        return_type_fmt += generate_return_types_named_tuple(
                            signature._ret._return_names, signature._name, block._name,
                        )
                        map_pairs.append(
                            f'    {{"{signature._name}", Get{signature._name}NamedTuple()}},'
                        )
                    schema_fmt += "  }\n"
                    i += 1
                schema_fmt += "  Py_RETURN_NONE;\n"
                schema_fmt += "  END_HANDLE_ERRORS\n"
                schema_fmt += "}\n"

        render_file_if_different(
            target_pybind_header_file, pybind_header_fmt.format(header_fmt)
        )
        render_file_if_different(
            target_pybind_source_file,
            pybind_source_fmt.format(
                schema_fmt, module_fmt, return_type_fmt, "\n".join(map_pairs)
            ),
        )
