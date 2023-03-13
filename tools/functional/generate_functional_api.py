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

from generator import Generator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--project_source_dir", type=str, help="The project source code directory.",
)
parser.add_argument(
    "--export_pybind",
    action="store_true",
    default=False,
    help="Whether to export pybind related files.",
)
args = parser.parse_args()

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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/layout.h"
#include "oneflow/core/framework/memory_format.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator.h"
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

#include "oneflow/core/functional/functional_api.yaml.h"
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

pybind_header_fmt = (
    license
    + """

#include <Python.h>

namespace oneflow {{
namespace one {{
namespace functional {{
{0}
}}  // namespace functional
}}  // namespace one
}}  // namespace oneflow
"""
)

pybind_source_fmt = (
    license
    + """

#include <Python.h>

#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/function_def.h"
#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/api/python/functional/python_arg_parser.h"
#include "oneflow/api/python/functional/python_return_types.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/extension/stack/python/stack_getter.h"

namespace {{
// This return type template code is referenced from:
// https://github.com/pytorch/pytorch/blob/master/tools/autograd/gen_python_functions.py
using oneflow::one::functional::returned_structseq_repr;
{2}

std::unordered_map<std::string, PyTypeObject*>& get_namedtuple_types_map() {{
  static std::unordered_map<std::string, PyTypeObject*> namedtuple_types_map = {{
{3}
  }};
  return namedtuple_types_map;
}}

PyTypeObject* get_namedtuple(const std::string& name) {{
  static auto& namedtuple_types_map = get_namedtuple_types_map();
  return namedtuple_types_map[name];
}}

}} // namespace


namespace oneflow {{
namespace one {{
namespace functional {{

PyObject* WrapTensorTuple(const TensorTuple& tensortuple,
                           const std::string& name) {{
  PyObjectPtr r(PyStructSequence_New(get_namedtuple(name)));
  if (!r) {{ throw py::error_already_set(); }}
  for (int i = 0; i < tensortuple.size(); ++i) {{
    PyTuple_SET_ITEM(r.get(), i, CastToPyObject(tensortuple[i]));
  }}
  return r.release();
}}
{0}
}}  // namespace functional
}}  // namespace one

namespace functional = one::functional;

ONEFLOW_API_PYBIND11_MODULE("_C", m) {{
  static PyMethodDef functions[] = {{
{1}
    {{NULL, NULL, 0, NULL}}
  }};

  PyObject* module = m.ptr();
  if (module) {{
    PyModule_AddFunctions(module, functions);
  }}
}}

}}  // namespace oneflow
"""
)

yaml_file_path = os.path.join(
    args.project_source_dir, "oneflow/core/functional/functional_api.yaml"
)
generated_api_dir = "oneflow/core/functional"
generated_pybind_dir = "oneflow/api/python/functional"

if __name__ == "__main__":
    assert os.path.isfile(yaml_file_path), (
        "It is not a regular file for the yaml file which is " + yaml_file_path
    )
    g = Generator(yaml_file_path)

    assert os.path.isdir(generated_api_dir), (
        "Could not locate the api generate directory which is " + generated_api_dir
    )
    target_header_file = os.path.join(generated_api_dir, "functional_api.yaml.h")
    g.generate_cpp_header_file(header_fmt, target_header_file)
    target_source_file = os.path.join(generated_api_dir, "functional_api.yaml.cpp")
    g.generate_cpp_source_file(source_fmt, target_source_file)
    if args.export_pybind:
        assert os.path.isdir(generated_pybind_dir), (
            "Could not locate the pybind generate directory which is "
            + generated_pybind_dir
        )
        target_pybind_header_file = os.path.join(
            generated_pybind_dir, "functional_api.yaml.pybind.h"
        )
        target_pybind_source_file = os.path.join(
            generated_pybind_dir, "functional_api.yaml.pybind.cpp"
        )
        g.generate_pybind_for_python(
            pybind_header_fmt,
            pybind_source_fmt,
            target_pybind_header_file,
            target_pybind_source_file,
        )
