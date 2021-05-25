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

#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_

#include <pybind11/pybind11.h>

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/functional/scalar.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

class PythonArg {
 public:
  PythonArg(py::object value) : value_(value.ptr()) {}

  virtual ~PythonArg() = default;

  operator int32_t() const { return py::cast<int32_t>(Borrow()); }
  operator uint32_t() const { return py::cast<uint32_t>(Borrow()); }
  operator int64_t() const { return py::cast<int64_t>(Borrow()); }
  operator uint64_t() const { return py::cast<uint64_t>(Borrow()); }
  operator float() const { return py::cast<float>(Borrow()); }
  operator double() const { return py::cast<double>(Borrow()); }
  operator bool() const { return py::cast<bool>(Borrow()); }
  operator std::string() const { return py::cast<std::string>(Borrow()); }

  operator std::vector<int32_t>() const;
  operator std::vector<uint32_t>() const;
  operator std::vector<int64_t>() const;
  operator std::vector<uint64_t>() const;
  operator std::vector<float>() const;
  operator std::vector<double>() const;
  operator std::vector<bool>() const;

  operator Scalar() const;

  operator std::shared_ptr<one::Tensor>() const {
    return py::cast<std::shared_ptr<one::Tensor>>(Borrow());
  }

  operator std::shared_ptr<one::TensorTuple>() const;
  operator one::TensorTuple() const { return *(std::shared_ptr<one::TensorTuple>(*this)); }

  operator std::shared_ptr<cfg::AttrValue>() const;

  operator AttrMap() const;

 private:
  py::object Borrow() const { return py::reinterpret_borrow<py::object>(value_); }
  PyObject* value_;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_
