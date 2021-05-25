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
#include <memory>
#include <pybind11/pybind11.h>
#include <glog/logging.h>

#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/framework/tensor.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

template<typename T>
std::vector<T> CastToList(py::object obj) {
  std::vector<T> values;
  if (detail::isinstance<py::list>(obj)) {
    py::list v_list = py::cast<py::list>(obj);
    values.resize(v_list.size());
    for (int i = 0; i < v_list.size(); ++i) { values[i] = py::cast<T>(v_list[i]); }
  } else if (detail::isinstance<py::tuple>(obj)) {
    py::tuple v_tuple = py::cast<py::tuple>(obj);
    values.resize(v_tuple.size());
    for (int i = 0; i < v_tuple.size(); ++i) { values[i] = py::cast<T>(v_tuple[i]); }
  } else {
    LOG(FATAL) << "The python object is not list or tuple.";
  }
  return values;
}

#define INSTANCE_CAST_TO_LIST(T) template std::vector<T> CastToList(py::object obj);

OF_PP_FOR_EACH_TUPLE(INSTANCE_CAST_TO_LIST, ARITHMETIC_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(std::string)
                                                OF_PP_MAKE_TUPLE_SEQ(std::shared_ptr<one::Tensor>));
#undef INSTANCE_CAST_TO_LIST

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow
