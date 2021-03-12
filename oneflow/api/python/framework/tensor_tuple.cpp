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
#include <pybind11/stl.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor.h"

namespace py = pybind11;

namespace oneflow {
namespace one {

namespace {

struct TensorTupleUtil final {
  static std::string ToString(const TensorTuple& tensor_tuple) {
    std::stringstream ss;
    int32_t idx = 0;
    ss << "TensorTuple(";
    for (const std::shared_ptr<Tensor>& tensor : tensor_tuple) {
      ss << tensor;
      if (++idx != tensor_tuple.size() || tensor_tuple.size() == 1) { ss << ", "; }
    }
    ss << ")";
    return ss.str();
  }

  static void MergeFrom(std::shared_ptr<TensorTuple>& tensor_tuple, const TensorTuple& other) {
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
      .def(py::init([](const std::shared_ptr<TensorTuple>& other) { return other; }))
      .def(py::init([](const std::vector<std::shared_ptr<Tensor>>& list) {
        auto tensor_tuple = std::make_shared<TensorTuple>();
        for (const auto& t : list) { tensor_tuple->emplace_back(t); }
        return tensor_tuple;
      }))
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
      .def("merge_from", &TensorTupleUtil::MergeFrom)
      .def("append", &TensorTupleUtil::AppendTensor);
}

}  // namespace one
}  // namespace oneflow
