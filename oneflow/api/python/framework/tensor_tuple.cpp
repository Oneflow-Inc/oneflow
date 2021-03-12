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

#include <vector>
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor.h"

namespace py = pybind11;

namespace oneflow {
namespace one {

namespace {

struct TensorTupleUtil final {
  static std::shared_ptr<TensorTuple> MakeTensorTupleByPyTuple(const py::tuple& py_tuple) {
    std::shared_ptr<TensorTuple> tensor_tuple = std::make_shared<TensorTuple>();
    for (const auto& tensor : py_tuple) {
      tensor_tuple->emplace_back(tensor.cast<std::shared_ptr<Tensor>>());
    }
    return tensor_tuple;
  }

  static std::shared_ptr<TensorTuple> MakeTensorTuple(const py::object& py_obj) {
    if (py::isinstance<TensorTuple>(py_obj)) {
      std::shared_ptr<TensorTuple> tensor_tuple = std::make_shared<TensorTuple>();
      for (const auto& tensor : py_obj) {
        tensor_tuple->emplace_back(tensor.cast<std::shared_ptr<Tensor>>());
      }
      return tensor_tuple;
    } else if (py::isinstance<py::tuple>(py_obj)) {
      return MakeTensorTupleByPyTuple(py_obj.cast<py::tuple>());
    } else if (py::isinstance<py::list>(py_obj)) {
      return MakeTensorTupleByPyTuple(py::tuple(py_obj.cast<py::list>()));
    } else {
      throw py::type_error("Input must be other TensorTuple, Tuple or List");
    }
  }

  static std::string ToString(const TensorTuple& tensor_tuple) {
    std::stringstream ss;
    int32_t idx = 0;
    ss << "TensorTuple(";
    for (const std::shared_ptr<Tensor>& tensor : tensor_tuple) {
      ss << tensor;
      if (++idx != tensor_tuple.size()) { ss << ", "; }
    }
    ss << ")";
    return ss.str();
  }

  static void AppendTensorTuple(std::shared_ptr<TensorTuple>& tensor_tuple,
                                const TensorTuple& other) {
    for (const auto& tensor : other) { tensor_tuple->emplace_back(tensor); }
  }

  static void AppendTensor(std::shared_ptr<TensorTuple>& tensor_tuple,
                           const std::shared_ptr<Tensor>& tensor) {
    tensor_tuple->push_back(tensor);
  }
};

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<TensorTuple, std::shared_ptr<TensorTuple>>(m, "TensorTuple")
      .def(py::init([]() { return std::make_shared<TensorTuple>(); }))
      .def(py::init(&TensorTupleUtil::MakeTensorTuple))
      .def("__str__", &TensorTupleUtil::ToString)
      .def("__repr__", &TensorTupleUtil::ToString)
      .def("__getitem__",
           [](const TensorTuple& tensor_tuple, int idx) { return tensor_tuple.at(idx); })
      .def("__setitem__",
           [](std::shared_ptr<TensorTuple>& tensor_tuple, int idx,
              const std::shared_ptr<Tensor>& tensor) { tensor_tuple->at(idx) = tensor; })
      .def(
          "__iter__",
          [](const TensorTuple& tensor_tuple) {
            return py::make_iterator(tensor_tuple.begin(), tensor_tuple.end());
          },
          py::keep_alive<0, 1>())
      .def("__len__", [](const TensorTuple& tensor_tuple) { return tensor_tuple.size(); })
      .def("append", &TensorTupleUtil::AppendTensorTuple)
      .def("append", &TensorTupleUtil::AppendTensor);
}

}  // namespace one
}  // namespace oneflow
