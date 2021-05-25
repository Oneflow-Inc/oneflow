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
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_

#include <string>
#include <vector>
#include <pybind11/pybind11.h>

#include "oneflow/core/common/preprocessor.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

#define ARITHMETIC_TYPE_SEQ      \
  OF_PP_MAKE_TUPLE_SEQ(int32_t)  \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t)  \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t) \
  OF_PP_MAKE_TUPLE_SEQ(float)    \
  OF_PP_MAKE_TUPLE_SEQ(double)   \
  OF_PP_MAKE_TUPLE_SEQ(bool)

#define MAKE_LIST_TUPLE_SEQ(T) (MAKE_LIST_TUPLE(T))
#define MAKE_LIST_TUPLE(T) (std::vector<T>)
#define ARITHMETIC_LIST_TYPE_SEQ OF_PP_FOR_EACH_TUPLE(MAKE_LIST_TUPLE_SEQ, ARITHMETIC_TYPE_SEQ)

template<typename T>
inline bool isinstance(py::object obj) {
  return py::isinstance<T>(obj);
}

#define IMPLEMENT_IS_INSTANCE(T)                     \
  template<>                                         \
  inline bool isinstance<T>(py::object obj) {        \
    static py::object dummy = py::cast(T());         \
    return py::isinstance(obj, py::type::of(dummy)); \
  }

OF_PP_FOR_EACH_TUPLE(IMPLEMENT_IS_INSTANCE, ARITHMETIC_TYPE_SEQ);
OF_PP_FOR_EACH_TUPLE(IMPLEMENT_IS_INSTANCE, ARITHMETIC_LIST_TYPE_SEQ);

IMPLEMENT_IS_INSTANCE(std::string);
IMPLEMENT_IS_INSTANCE(std::vector<std::string>);
#undef IMPLEMENT_IS_INSTANCE

template<typename T>
std::vector<T> CastToList(py::object obj);

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_
