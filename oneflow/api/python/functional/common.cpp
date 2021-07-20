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
#include "oneflow/core/functional/functional.h"

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

Maybe<DataType> InferScalarType(PyObject* object) {
  if (PyLong_Check(object)) {
    return DataType::kInt64;
  } else if (PyBool_Check(object)) {
    return DataType::kUInt8;
  } else if (PySequence_Check(object)) {
    int64_t length = PySequence_Length(object);
    CHECK_GT_OR_RETURN(length, 0) << "Failed to get sequence index length.";
    DataType scalar_type = DataType::kInvalidDataType;
    for (int64_t i = 0; i < length; ++i) {
      PyObjectPtr item(PySequence_GetItem(object, i));
      const auto& item_scalar_type = JUST(InferScalarType(item.get()));
      if (scalar_type != DataType::kInvalidDataType) {
        CHECK_EQ_OR_RETURN(scalar_type, item_scalar_type)
            << "Different scalar types are not allowed.";
      } else {
        scalar_type = item_scalar_type;
      }
    }
    return scalar_type;
  }
  UNIMPLEMENTED_THEN_RETURN() << "Could not infer scalar type of " << Py_TYPE(object)->tp_name;
}

Maybe<Tensor> CastToIndexingTensor(PyObject* object) {
  // TODO(hjchen2): Support lazy.
  DataType dtype = JUST(InferScalarType(object));
  DimVector sizes;
  PyObject* seq = object;
  PyObjectPtr handle;
  while (PySequence_Check(seq)) {
    int64_t length = PySequence_Length(seq);
    CHECK_GT_OR_RETURN(length, 0) << "Failed to get sequence index length.";
    sizes.push_back(length);
    CHECK_LE_OR_RETURN(sizes.size(), /*MAX_DIMS=*/128)
        << "Too many dimensions " << Py_TYPE(seq)->tp_name;
    if (length == 0) break;
    handle = PyObjectPtr(PySequence_GetItem(seq, 0));
    seq = handle.get();
  }
  Shape shape(sizes);
  const auto& tensor = JUST(functional::Constant(shape, 0, dtype));
  CHECK_OR_RETURN(JUST(tensor->has_eager_blob_object()))
      << "Only converting to eager tensor is valid, please check whether the op is running in "
         "eager mode or not.";
  return tensor;
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
  } else if (PySequence_Check(object)) {
    return std::make_shared<detail::IndexItem>(JUST(detail::CastToIndexingTensor(object)));
  }
  UNIMPLEMENTED_THEN_RETURN() << "Invalid index of " << Py_TYPE(object)->tp_name;
}

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow
