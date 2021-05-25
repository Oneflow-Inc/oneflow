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

#include "oneflow/api/python/functional/python_arg.h"

#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

#define IMPLICIT_TRANSFORM_LIST_OP(T) \
  PythonArg::operator std::vector<T>() const { return detail::CastToList<T>(Borrow()); }

OF_PP_FOR_EACH_TUPLE(IMPLICIT_TRANSFORM_LIST_OP,
                     ARITHMETIC_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(std::string));
#undef IMPLICIT_TRANSFORM_LIST_OP

PythonArg::operator Scalar() const {
  py::object obj = Borrow();
  if (detail::isinstance<int32_t>(obj)) {
    return Scalar(py::cast<int32_t>(obj));
  } else if (detail::isinstance<int64_t>(obj)) {
    return Scalar(py::cast<int64_t>(obj));
  } else if (detail::isinstance<float>(obj)) {
    return Scalar(py::cast<float>(obj));
  } else if (detail::isinstance<double>(obj)) {
    return Scalar(py::cast<double>(obj));
  } else if (detail::isinstance<bool>(obj)) {
    return Scalar(py::cast<bool>(obj));
  } else {
    UNIMPLEMENTED() << "Can not convert to scalar from python object with type "
                    << py::cast<std::string>(py::str(py::type::of(obj)));
    return Scalar(0);
  }
}

PythonArg::operator std::shared_ptr<one::TensorTuple>() const {
  py::object obj = Borrow();
  if (detail::isinstance<one::TensorTuple>(obj)) {
    return py::cast<std::shared_ptr<one::TensorTuple>>(obj);
  }
  auto v = detail::CastToList<std::shared_ptr<one::Tensor>>(obj);
  auto values = std::make_shared<one::TensorTuple>(v.size());
  for (int i = 0; i < v.size(); ++i) { values->at(i) = v[i]; }
  return values;
}

PythonArg::operator std::shared_ptr<cfg::AttrValue>() const {
  // TODO()
  py::object obj = Borrow();
  if (detail::isinstance<cfg::AttrValue>(obj)) {
    return py::cast<std::shared_ptr<cfg::AttrValue>>(obj);
  }
  auto attr_value = std::make_shared<cfg::AttrValue>();
  if (detail::isinstance<int32_t>(obj)) {
    attr_value->set_at_int32(py::cast<int32_t>(obj));
  } else if (detail::isinstance<double>(obj)) {
    attr_value->set_at_double(py::cast<double>(obj));
  } else {
    LOG(FATAL) << "The attribute type was not supported which is "
               << py::cast<std::string>(py::str(py::type::of(obj)));
  }
  return attr_value;
}

PythonArg::operator AttrMap() const {
  py::object obj = Borrow();
  CHECK(detail::isinstance<MutableCfgAttrMap>(obj));
  return *(py::cast<std::shared_ptr<MutableCfgAttrMap>>(obj));
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
