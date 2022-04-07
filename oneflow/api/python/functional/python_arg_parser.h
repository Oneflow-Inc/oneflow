/*
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
*/
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_PARSER_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_PARSER_H_

#include <pybind11/pybind11.h>
#include <Python.h>

#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/function_def.h"
#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace one {
namespace functional {

template<int N>
class ParsedArgs {
 public:
  ParsedArgs() = default;

  const PythonArg& operator[](size_t idx) const {
    CHECK_LT_OR_THROW(idx, N) << "index " << idx << " is out of bound [0, " << N << ")";
    return data[idx];
  }

  PythonArg& operator[](size_t idx) {
    CHECK_LT_OR_THROW(idx, N) << "index " << idx << " is out of bound [0, " << N << ")";
    return data[idx];
  }

 public:
  PythonArg data[N];
};

class FunctionSignature {
 public:
  FunctionSignature() = default;
  FunctionSignature(const std::string& signature, const FunctionDef* def, size_t max_pos_nargs)
      : signature_(signature), def_(def), max_pos_nargs_(max_pos_nargs) {}

  const std::string& signature() const { return signature_; }

  bool Parse(PyObject* args, PyObject* kwargs, PythonArg* parsed_args, bool raise_exception) const;

 private:
  void ReportKwargsError(PyObject* kwargs) const;

  std::string signature_;
  const FunctionDef* def_;
  size_t max_pos_nargs_;
};

template<typename... SchemaT>
class PythonArgParser {
 public:
  static_assert(sizeof...(SchemaT) >= 1, "requires 1 template argument at least.");
  static constexpr size_t kSchemaSize = sizeof...(SchemaT);
  static constexpr size_t N = std::max({SchemaT::max_args...});

  template<size_t I>
  using schema_t = typename std::tuple_element<I, std::tuple<SchemaT...>>::type;

  PythonArgParser(const std::string& name) : name_(name) {
    Init(std::make_index_sequence<sizeof...(SchemaT)>{});
  }

  int Parse(PyObject* args, PyObject* kwargs, ParsedArgs<N>* parsed_args) const {
    bool raise_exception = (kSchemaSize == 1);
    for (int i = 0; i < kSchemaSize; ++i) {
      if (signatures_[i].Parse(args, kwargs, parsed_args->data, raise_exception)) { return i; }
    }
    ReportInvalidArgsError(args, kwargs);
    return -1;
  }

 private:
  template<size_t... I>
  void Init(std::index_sequence<I...>) {
    __attribute__((__unused__)) int dummy[] = {
        ((void)(signatures_[I] = FunctionSignature(
                    schema_t<I>::signature, &schema_t<I>::function_def, schema_t<I>::max_pos_args)),
         0)...};
  }

  void ReportInvalidArgsError(PyObject* args, PyObject* kwargs) const {
    std::ostringstream ss;
    ss << name_ << "(): received an invalid combination of arguments. The valid signatures are:";
    for (int i = 0; i < kSchemaSize; ++i) {
      ss << "\n\t*" << i << ": " << signatures_[i].signature();
    }
    THROW(TypeError) << ss.str();
  }

  std::string name_;
  FunctionSignature signatures_[kSchemaSize];
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_PARSER_H_
