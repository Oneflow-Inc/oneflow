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
#include "oneflow/api/python/functional/indexing.h"

#include <object.h>
#include <pybind11/pybind11.h>
#include "oneflow/api/python/functional/common.h"
#include "oneflow/extension/python/numpy.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/api/python/functional/tensor_api.yaml.h"
#include "oneflow/core/common/foreign_lock_helper.h"

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

void PySliceUnpack(PyObject* object, Py_ssize_t* start, Py_ssize_t* stop, Py_ssize_t* step) {
  PySliceObject* obj = (PySliceObject*)object;
  if (obj->step == Py_None) {
    *step = 1;
  } else {
    CHECK_OR_THROW(_PyEval_SliceIndex(obj->step, step))
        << "Invalid slice " << PyObjectToReprStr(object);
    CHECK_NE_OR_THROW(*step, 0) << "slice step cannot be zero.";
    if (*step < -PY_SSIZE_T_MAX) *step = -PY_SSIZE_T_MAX;
  }
  if (obj->start == Py_None) {
    *start = *step < 0 ? PY_SSIZE_T_MAX : 0;
  } else {
    CHECK_OR_THROW(_PyEval_SliceIndex(obj->start, start))
        << "Invalid slice " << PyObjectToReprStr(object);
  }
  if (obj->stop == Py_None) {
    *stop = *step < 0 ? PY_SSIZE_T_MIN : PY_SSIZE_T_MAX;
  } else {
    CHECK_OR_THROW(_PyEval_SliceIndex(obj->stop, stop))
        << "Invalid slice " << PyObjectToReprStr(object);
  }
}

DataType InferScalarType(PyObject* object) {
  if (PyBool_Check(object)) {
    return DataType::kBool;
  } else if (PyLong_Check(object)) {
    return DataType::kInt64;
  } else if (PyArray_Check(object)) {
    return numpy::GetOFDataTypeFromNpArray(reinterpret_cast<PyArrayObject*>(object)).GetOrThrow();
  } else if (PyArray_CheckScalar(object)) {
    return numpy::NumpyTypeToOFDataType(PyArray_DescrFromScalar(object)->type_num).GetOrThrow();
  } else if (PySequence_Check(object)) {
    int64_t length = PySequence_Length(object);
    if (length == 0) { return DataType::kInt64; }
    DataType scalar_type = DataType::kInvalidDataType;
    for (int64_t i = 0; i < length; ++i) {
      PyObjectPtr item(PySequence_GetItem(object, i));
      const auto& item_scalar_type = InferScalarType(item.get());
      if (scalar_type != DataType::kInvalidDataType) {
        CHECK_EQ_OR_THROW(scalar_type, item_scalar_type)
            << "Different scalar types are not allowed.";
      } else {
        scalar_type = item_scalar_type;
      }
    }
    return scalar_type;
  }
  THROW(TypeError) << "Can't infer scalar type of " << Py_TYPE(object)->tp_name;
  return DataType::kInvalidDataType;
}

void ParseScalar(PyObject* object, char* data, const DataType& dtype) {
  if (dtype == DataType::kInt64) {
    CHECK_OR_THROW(PyLong_Check(object) || numpy::PyArrayCheckLongScalar(object))
        << "Expected a long value.";
    *(reinterpret_cast<int64_t*>(data)) = PyLong_AsLongLong(object);
  } else if (dtype == DataType::kInt32) {
    CHECK_OR_THROW(PyLong_Check(object) || numpy::PyArrayCheckLongScalar(object))
        << "Expected a long value.";
    *(reinterpret_cast<int32_t*>(data)) = PyLong_AsLongLong(object);
  } else if (dtype == DataType::kUInt8 || dtype == DataType::kBool) {
    CHECK_OR_THROW(PyBool_Check(object) || PyLong_Check(object)
                   || numpy::PyArrayCheckLongScalar(object))
        << "Expected a boolean or long value.";
    if (PyBool_Check(object) || numpy::PyArrayCheckBoolScalar(object)) {
      *(reinterpret_cast<bool*>(data)) = (object == Py_True);
    } else {
      int64_t value = PyLong_AsLongLong(object);
      CHECK_OR_THROW(value >= 0 && value <= 255) << "Out of range 0-255.";
      *(reinterpret_cast<uint8_t*>(data)) = static_cast<uint8_t>(value);
    }
  } else {
    THROW(TypeError) << "Can't parse scalar with data type " << dtype;
  }
}

void RecursiveParseAndAssign(PyObject* object, char* data, const int& ndims, const int& dim,
                             const ShapeView& shape, const DimVector& strides,
                             const DataType& dtype) {
  if (dim == ndims) { return ParseScalar(object, data, dtype); }
  auto seq = PyObjectPtr(PySequence_Fast(object, "Expected a sequence."));
  int64_t size = PySequence_Fast_GET_SIZE(seq.get());
  CHECK_EQ_OR_THROW(size, shape.At(dim)) << "Sequence size is " << size << " at dimemsion " << dim
                                         << ", but expected " << shape.At(dim);
  for (int64_t i = 0; i < size; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq.get(), i);
    RecursiveParseAndAssign(item, data, ndims, dim + 1, shape, strides, dtype);
    data += strides.at(dim) * GetSizeOfDataType(dtype);
  }
}

void ParseArrayToTensor(PyObject* object,
                        const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
  const DataType dtype = eager_blob_object->data_type();
  const int ndims = eager_blob_object->shape().NumAxes();
  DimVector strides(ndims);
  int64_t size = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = size;
    size *= eager_blob_object->shape().At(i);
  }
  RecursiveParseAndAssign(object, eager_blob_object->mut_dptr<char>(), ndims, 0,
                          eager_blob_object->shape(), strides, dtype);
}

Shape InferArraySizes(PyObject* object) {
  DimVector sizes;
  PyObject* seq = object;
  PyObjectPtr handle;
  while (PySequence_Check(seq)) {
    int64_t length = PySequence_Length(seq);
    sizes.emplace_back(length);
    CHECK_LE_OR_THROW(sizes.size(), /*MAX_DIMS=*/128)
        << "Too many dimensions " << Py_TYPE(seq)->tp_name;
    if (length == 0) break;
    handle = PyObjectPtr(PySequence_GetItem(seq, 0));
    seq = handle.get();
  }
  return Shape(sizes);
}

Maybe<Tensor> ConvertToIndexingTensor(PyObject* object) {
  // NOTE: convert data to indexing will ensure in eager mode
  LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
  const DataType dtype = InferScalarType(object);
  const auto& device = JUST(Device::New("cpu"));

  // index type must be integers
  if (!(IsIntegralDataType(dtype) || (IsBoolDataType(dtype)))) {
    return Error::IndexError() << "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                                  "(`None`) and integer or boolean arrays are valid indices";
  }
  // In advanced indexing condition, index can be array object, need to handle it specially.
  if (PyArray_Check(object)) {
    return TensorWithData(object, NullOpt, device, /*requires_grad=*/false, /*pin_memory=*/false);
  }

  const auto& sizes = InferArraySizes(object);
  const auto& tensor = JUST(functional::Empty(sizes, CHECK_JUST(DType::Get(dtype)), device,
                                              /*requires_grad=*/false, /*pin_memory=*/false));
  // Prevent the python object release until the callback is complete.
  Py_INCREF(object);
  auto handle = std::shared_ptr<PyObject>(PyObjectPtr(object));

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->AccessBlobByCallback(
        JUST(tensor->AsLocalTensor()),
        [handle](ep::Stream* stream,
                 const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
          CHECK_JUST(Singleton<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
            ParseArrayToTensor(handle.get(), eager_blob_object);
            return Maybe<void>::Ok();
          }));
        },
        "mut");
  }));
  return tensor;
}

IndexItem UnpackIndexItem(PyObject* object) {
  if (object == Py_Ellipsis) {
    return IndexItem(EllipsisIndex{});
  } else if (PySlice_Check(object)) {
    Py_ssize_t start, end, step;
    PySliceUnpack(object, &start, &end, &step);
    return IndexItem(start, end, step);
  } else if (PyLong_Check(object) && object != Py_False && object != Py_True) {
    return IndexItem(static_cast<int64_t>(PyLong_AsLongLong(object)));
  } else if (numpy::PyArrayCheckLongScalar(object)) {
    return IndexItem(static_cast<int64_t>(PyLong_AsLongLong(object)));
  } else if (object == Py_False || object == Py_True) {
    return IndexItem(object == Py_True);
  } else if (object == Py_None) {
    return IndexItem(NoneIndex{});
  } else if (PyTensor_Check(object)) {
    return IndexItem(PyTensor_Unpack(object));
  } else if (PySequence_Check(object)) {
    return IndexItem(ConvertToIndexingTensor(object).GetPtrOrThrow());
  }
  THROW(IndexError) << "Invalid index " << Py_TYPE(object)->tp_name;
  return IndexItem();
}

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow
