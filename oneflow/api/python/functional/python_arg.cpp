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

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

PythonArg::operator std::shared_ptr<one::TensorTuple>() const {
  py::object obj = Borrow();
  if (detail::isinstance<one::TensorTuple>(obj)) {
    return py::cast<std::shared_ptr<one::TensorTuple>>(Borrow());
  }
  CHECK(detail::isinstance<py::list>(obj));
  py::list list = py::cast<py::list>(obj);
  auto tensor_tuple = std::make_shared<one::TensorTuple>(list.size());
  for (int i = 0; i < list.size(); ++i) {
    CHECK(detail::isinstance<one::Tensor>(list[i]));
    tensor_tuple->at(i) = py::cast<std::shared_ptr<one::Tensor>>(list[i]);
  }
  return tensor_tuple;
}

PythonArg::operator std::shared_ptr<cfg::AttrValue>() const {
  // TODO()
  py::object obj = Borrow();
  if (detail::isinstance<cfg::AttrValue>(obj)) {
    return py::cast<std::shared_ptr<cfg::AttrValue>>(obj);
  }
  auto attr_value = std::make_shared<cfg::AttrValue>();
  if (detail::isinstance<int>(obj)) {
    attr_value->set_at_int32(py::cast<int>(obj));
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
