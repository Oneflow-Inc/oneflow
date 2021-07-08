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

#include "oneflow/api/python/functional/common.h"

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

Maybe<void> PySliceUnpack(PyObject* object, Py_ssize_t* start, Py_ssize_t* stop, Py_ssize_t* step) {
  PySliceObject* obj = (PySliceObject*)object;
  if (obj->step == Py_None) {
    *step = 1;
  } else {
    CHECK_OR_RETURN(_PyEval_SliceIndex(obj->step, step))
        << "Invalid slice " << PyStringAsString(PyObject_Repr(object));
    CHECK_NE_OR_RETURN(*step, 0) << "slice step cannot be zero.";
    if (*step < -PY_SSIZE_T_MAX) *step = -PY_SSIZE_T_MAX;
  }

  if (obj->start == Py_None) {
    *start = *step < 0 ? PY_SSIZE_T_MAX : 0;
  } else {
    CHECK_OR_RETURN(_PyEval_SliceIndex(obj->start, start))
        << "Invalid slice " << PyStringAsString(PyObject_Repr(object));
  }

  if (obj->stop == Py_None) {
    *stop = *step < 0 ? PY_SSIZE_T_MIN : PY_SSIZE_T_MAX;
  } else {
    CHECK_OR_RETURN(_PyEval_SliceIndex(obj->stop, stop))
        << "Invalid slice " << PyStringAsString(PyObject_Repr(object));
  }
  return Maybe<void>::Ok();
}

const char* PyStringAsString(PyObject* object) {
  return PyBytes_AsString(PyUnicode_AsEncodedString(object, "utf-8", "~E~"));
}

Maybe<detail::IndexItem> UnpackIndexItem(PyObject* object) {
  if (object == Py_Ellipsis) {
    return std::make_shared<detail::IndexItem>(detail::EllipsisIndex{});
  } else if (PySlice_Check(object)) {
    Py_ssize_t start, end, step;
    JUST(PySliceUnpack(object, &start, &end, &step));
    return std::make_shared<detail::IndexItem>(start, end, step);
  } else if (PyLong_Check(object) && object != Py_False && object != Py_True) {
    return std::make_shared<detail::IndexItem>(static_cast<int64_t>(PyLong_AsLongLong(object)));
  } else if (object == Py_False || object == Py_True) {
    return std::make_shared<detail::IndexItem>(object == Py_True);
  } else if (object == Py_None) {
    return std::make_shared<detail::IndexItem>(detail::NoneIndex{});
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "Invalid index " << PyStringAsString(PyObject_Repr(object));
  }
}

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow
