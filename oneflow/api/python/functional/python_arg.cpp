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

#include "oneflow/api/python/framework/tensor.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/indexing.h"
#include "oneflow/extension/python/numpy.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/layout.h"
#include "oneflow/core/framework/memory_format.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/tensor_index.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

#define INSTANCE_OBJECT_AS_INTEGER(T)                                                            \
  template<>                                                                                     \
  T PythonArg::ObjectAs<T>() const {                                                             \
    if (PyIntegerScalarTensorCheck(object_)) {                                                   \
      return static_cast<T>(PyUnpackIntegerScalarTensor_AsLongLong(object_));                    \
    }                                                                                            \
    return static_cast<T>(PyLong_AsLongLong(object_));                                           \
  }                                                                                              \
  template<>                                                                                     \
  std::vector<T> PythonArg::ObjectAs<std::vector<T>>() const {                                   \
    if (size_ > 0 && PyLong_Check(object_)) {                                                    \
      return std::vector<T>(size_, static_cast<T>(PyLong_AsLongLong(object_)));                  \
    }                                                                                            \
    return PyUnpackLongSequence<T>(object_);                                                     \
  }                                                                                              \
  template<>                                                                                     \
  std::shared_ptr<std::vector<T>> PythonArg::ObjectAs<std::shared_ptr<std::vector<T>>>() const { \
    return std::make_shared<std::vector<T>>(ObjectAs<std::vector<T>>());                         \
  }

OF_PP_FOR_EACH_TUPLE(INSTANCE_OBJECT_AS_INTEGER, INTEGER_AND_BOOL_TYPE_SEQ)
#undef INSTANCE_OBJECT_AS_INTEGER

#define INSTANCE_OBJECT_AS_FLOAT(T)                                                              \
  template<>                                                                                     \
  T PythonArg::ObjectAs<T>() const {                                                             \
    if (PyFloatScalarTensorCheck(object_)) {                                                     \
      return static_cast<T>(PyUnpackFloatScalarTensor_AsDouble(object_));                        \
    }                                                                                            \
    return static_cast<T>(PyFloat_AsDouble(object_));                                            \
  }                                                                                              \
  template<>                                                                                     \
  std::vector<T> PythonArg::ObjectAs<std::vector<T>>() const {                                   \
    if (size_ > 0 && PyFloat_Check(object_)) {                                                   \
      return std::vector<T>(size_, static_cast<T>(PyFloat_AsDouble(object_)));                   \
    }                                                                                            \
    return PyUnpackFloatSequence<T>(object_);                                                    \
  }                                                                                              \
  template<>                                                                                     \
  std::shared_ptr<std::vector<T>> PythonArg::ObjectAs<std::shared_ptr<std::vector<T>>>() const { \
    return std::make_shared<std::vector<T>>(ObjectAs<std::vector<T>>());                         \
  }

OF_PP_FOR_EACH_TUPLE(INSTANCE_OBJECT_AS_FLOAT, FLOATING_TYPE_SEQ)
#undef INSTANCE_OBJECT_AS_FLOAT

#define INSTANCE_OBJECT_AS_SHARED_PTR(T)                               \
  template<>                                                           \
  std::shared_ptr<T> PythonArg::ObjectAs<std::shared_ptr<T>>() const { \
    return std::make_shared<T>(ObjectAs<T>());                         \
  }

template<>
std::string PythonArg::ObjectAs<std::string>() const {
  return PyStringAsString(object_);
}
INSTANCE_OBJECT_AS_SHARED_PTR(std::string)

template<>
Scalar PythonArg::ObjectAs<Scalar>() const {
  if (PyScalarTensorCheck(object_)) { return PyUnpackScalarTensor(object_); }
  return PyUnpackScalar(object_);
}
INSTANCE_OBJECT_AS_SHARED_PTR(Scalar)

template<>
std::shared_ptr<one::Tensor> PythonArg::ObjectAs<std::shared_ptr<one::Tensor>>() const {
  return PyTensor_Unpack(object_);
}

template<>
one::TensorTuple PythonArg::ObjectAs<one::TensorTuple>() const {
  if (PyTensorTupleCheck(object_)) { return *PyUnpackTensorTuple(object_); }
  const auto& v = PyUnpackTensorSequence(object_);
  one::TensorTuple values(v.size());
  for (int i = 0; i < v.size(); ++i) { values[i] = v.at(i); }
  return values;
}
INSTANCE_OBJECT_AS_SHARED_PTR(one::TensorTuple)

template<>
Symbol<DType> PythonArg::ObjectAs<Symbol<DType>>() const {
  return PyUnpackDType(object_);
}

template<>
Symbol<Layout> PythonArg::ObjectAs<Symbol<Layout>>() const {
  return PyUnpackLayout(object_);
}

template<>
Symbol<MemoryFormat> PythonArg::ObjectAs<Symbol<MemoryFormat>>() const {
  return PyUnpackMemoryFormat(object_);
}

template<>
std::vector<Symbol<DType>> PythonArg::ObjectAs<std::vector<Symbol<DType>>>() const {
  return PyUnpackDTypeSequence(object_);
}
INSTANCE_OBJECT_AS_SHARED_PTR(std::vector<Symbol<DType>>)

template<>
Shape PythonArg::ObjectAs<Shape>() const {
  return PyUnpackShape(object_);
}
INSTANCE_OBJECT_AS_SHARED_PTR(Shape)

template<>
std::vector<Shape> PythonArg::ObjectAs<std::vector<Shape>>() const {
  return PyUnpackShapeSequence(object_);
}
INSTANCE_OBJECT_AS_SHARED_PTR(std::vector<Shape>)

template<>
std::shared_ptr<one::Generator> PythonArg::ObjectAs<std::shared_ptr<one::Generator>>() const {
  return PyUnpackGenerator(object_);
}

template<>
Symbol<Device> PythonArg::ObjectAs<Symbol<Device>>() const {
  if (PyStringCheck(object_)) {
    std::string device_str = PyStringAsString(object_);
    return Device::ParseAndNew(device_str).GetOrThrow();
  }
  return PyUnpackDevice(object_);
}

template<>
Symbol<ParallelDesc> PythonArg::ObjectAs<Symbol<ParallelDesc>>() const {
  return PyUnpackParallelDesc(object_);
}

template<>
Symbol<SbpParallel> PythonArg::ObjectAs<Symbol<SbpParallel>>() const {
  return PyUnpackSbpParallel(object_);
}

template<>
std::vector<Symbol<SbpParallel>> PythonArg::ObjectAs<std::vector<Symbol<SbpParallel>>>() const {
  if (PySbpParallelCheck(object_)) {
    return std::vector<Symbol<SbpParallel>>(1, PyUnpackSbpParallel(object_));
  }
  return PyUnpackSbpParallelSequence(object_);
}
INSTANCE_OBJECT_AS_SHARED_PTR(std::vector<Symbol<SbpParallel>>)

template<>
TensorIndex PythonArg::ObjectAs<TensorIndex>() const {
  return PyUnpackTensorIndex(object_);
}
INSTANCE_OBJECT_AS_SHARED_PTR(TensorIndex)

template<>
std::shared_ptr<one::OpExpr> PythonArg::ObjectAs<std::shared_ptr<one::OpExpr>>() const {
  return PyUnpackOpExpr(object_);
}

template<>
PyObject* PythonArg::ObjectAs<PyObject*>() const {
  return object_;
}

template<>
std::vector<std::string> PythonArg::ObjectAs<std::vector<std::string>>() const {
  return PyUnpackSequence<std::string>(
      object_, [](PyObject* item) -> std::string { return PyStringAsString(item); });
}

INSTANCE_OBJECT_AS_SHARED_PTR(std::vector<std::string>)

#undef INSTANCE_OBJECT_AS_SHARED_PTR

bool PythonArg::TypeCheck(ValueType type) const {
  if (tag_ == HAS_DEFAULT) { return default_val_->value_type() == type; }
  switch (type) {
    case kINT32:
    case kUINT32:
    case kINT64:
    case kUINT64:
    case kBOOL:
      return PyLong_Check(object_) || numpy::PyArrayCheckLongScalar(object_)
             || PyIntegerScalarTensorCheck(object_) || PyBoolScalarTensorCheck(object_);
    case kINT32_LIST:
    case kUINT32_LIST:
    case kINT64_LIST:
    case kUINT64_LIST:
    case kBOOL_LIST: return PyLongSequenceCheck(object_) || (size_ > 0 && PyLong_Check(object_));
    case kFLOAT:
    case kDOUBLE:
      return PyFloat_Check(object_) || PyLong_Check(object_)
             || numpy::PyArrayCheckFloatScalar(object_) || numpy::PyArrayCheckLongScalar(object_)
             || PyFloatScalarTensorCheck(object_) || PyIntegerScalarTensorCheck(object_);
    case kFLOAT_LIST:
    case kDOUBLE_LIST:
      return PyFloatSequenceCheck(object_)
             || (size_ > 0 && (PyFloat_Check(object_) || PyLong_Check(object_)));
    case kSTRING: return PyStringCheck(object_);
    case kSTRING_LIST: return PyStringSequenceCheck(object_);
    case kSCALAR:
      return PyScalarCheck(object_) || numpy::PyArrayCheckLongScalar(object_)
             || numpy::PyArrayCheckFloatScalar(object_) || PyScalarTensorCheck(object_);
    case kTENSOR:
    case kTENSOR_REF: return PyTensor_Check(object_);
    case kTENSOR_TUPLE: return PyTensorTupleCheck(object_) || PyTensorSequenceCheck(object_);
    case kDTYPE: return PyDTypeCheck(object_);
    case kLAYOUT: return PyLayoutCheck(object_);
    case kMEMORYFORMAT: return PyMemoryFormatCheck(object_);
    case kSHAPE: return PyLongSequenceCheck(object_);
    case kGENERATOR:
    case kGENERATOR_REF: return PyGeneratorCheck(object_);
    case kTENSOR_INDEX: return PyTensorIndexCheck(object_);
    case kDEVICE: return PyStringCheck(object_) || PyDeviceCheck(object_);
    case kPARALLEL_DESC: return PyParallelDescCheck(object_);
    case kSBP_PARALLEL: return PySbpParallelCheck(object_);
    case kSBP_PARALLEL_LIST:
      return PySbpParallelSequenceCheck(object_) || PySbpParallelCheck(object_);
    case kOPEXPR_REF: return PyOpExprCheck(object_);
    case kPY_OBJECT: return nullptr != object_;
    case kDTYPE_LIST: return PyDTypeSequenceCheck(object_);
    case kSHAPE_LIST: return PyShapeSequenceCheck(object_);
    case kCOMPLEX_FLOAT:
    case kCOMPLEX_DOUBLE:
      return PyComplex_Check(object_) || PyFloat_Check(object_) || PyLong_Check(object_)
             || numpy::PyArrayCheckComplexScalar(object_) || numpy::PyArrayCheckFloatScalar(object_)
             || numpy::PyArrayCheckLongScalar(object_) || PyComplexScalarTensorCheck(object_)
             || PyFloatScalarTensorCheck(object_) || PyIntegerScalarTensorCheck(object_);
    default: {
      THROW(RuntimeError) << "Can not check type " << ValueTypeName(type);
    }
  }
  return false;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
