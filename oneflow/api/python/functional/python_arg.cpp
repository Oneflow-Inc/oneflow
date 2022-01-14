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
#include "oneflow/api/python/functional/indexing.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/dtype.h"
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

#define INSTANCE_OBJECT_AS_INTEGER(T)                                                             \
  template<>                                                                                      \
  Maybe<T> PythonArg::ObjectAs<T>() const {                                                       \
    return static_cast<T>(PyLong_AsLongLong(object_));                                            \
  }                                                                                               \
  template<>                                                                                      \
  Maybe<std::vector<T>> PythonArg::ObjectAs<std::vector<T>>() const {                             \
    if (size_ > 0 && PyLong_Check(object_)) {                                                     \
      return std::make_shared<std::vector<T>>(size_, static_cast<T>(PyLong_AsLongLong(object_))); \
    }                                                                                             \
    return PyUnpackLongSequence<T>(object_);                                                      \
  }

OF_PP_FOR_EACH_TUPLE(INSTANCE_OBJECT_AS_INTEGER, INTEGER_TYPE_SEQ)
#undef INSTANCE_OBJECT_AS_INTEGER

#define INSTANCE_OBJECT_AS_FLOAT(T)                                                              \
  template<>                                                                                     \
  Maybe<T> PythonArg::ObjectAs<T>() const {                                                      \
    return static_cast<T>(PyFloat_AsDouble(object_));                                            \
  }                                                                                              \
  template<>                                                                                     \
  Maybe<std::vector<T>> PythonArg::ObjectAs<std::vector<T>>() const {                            \
    if (size_ > 0 && PyFloat_Check(object_)) {                                                   \
      return std::make_shared<std::vector<T>>(size_, static_cast<T>(PyFloat_AsDouble(object_))); \
    }                                                                                            \
    return PyUnpackFloatSequence<T>(object_);                                                    \
  }

OF_PP_FOR_EACH_TUPLE(INSTANCE_OBJECT_AS_FLOAT, FLOATING_TYPE_SEQ)
#undef INSTANCE_OBJECT_AS_FLOAT

template<>
Maybe<std::string> PythonArg::ObjectAs<std::string>() const {
  return JUST(PyStringAsString(object_));
}

template<>
Maybe<Scalar> PythonArg::ObjectAs<Scalar>() const {
  return PyUnpackScalar(object_);
}

template<>
Maybe<std::shared_ptr<one::Tensor>> PythonArg::ObjectAs<std::shared_ptr<one::Tensor>>() const {
  return JUST(PyUnpackTensor(object_));
}

template<>
Maybe<one::Tensor> PythonArg::ObjectAs<one::Tensor>() const {
  return PyUnpackTensor(object_);
}

template<>
Maybe<std::shared_ptr<one::TensorTuple>> PythonArg::ObjectAs<std::shared_ptr<one::TensorTuple>>()
    const {
  if (PyTensorTupleCheck(object_)) { return JUST(PyUnpackTensorTuple(object_)); }
  const auto& v = JUST(PyUnpackTensorSequence(object_));
  auto values = std::make_shared<one::TensorTuple>(v->size());
  for (int i = 0; i < v->size(); ++i) { values->at(i) = v->at(i); }
  return values;
}

template<>
Maybe<one::TensorTuple> PythonArg::ObjectAs<one::TensorTuple>() const {
  return *JUST(ObjectAs<std::shared_ptr<one::TensorTuple>>());
}

template<>
Maybe<Symbol<DType>> PythonArg::ObjectAs<Symbol<DType>>() const {
  return PyUnpackDType(object_);
}

template<>
Maybe<std::vector<Symbol<DType>>> PythonArg::ObjectAs<std::vector<Symbol<DType>>>() const {
  return PyUnpackDTypeSequence(object_);
}

template<>
Maybe<Shape> PythonArg::ObjectAs<Shape>() const {
  const auto& shape = JUST(PyUnpackLongSequence<int64_t>(object_));
  return std::make_shared<Shape>(DimVector(shape->begin(), shape->end()));
}

template<>
Maybe<std::vector<Shape>> PythonArg::ObjectAs<std::vector<Shape>>() const {
  return PyUnpackShapeSequence(object_);
}

template<>
Maybe<std::shared_ptr<one::Generator>> PythonArg::ObjectAs<std::shared_ptr<one::Generator>>()
    const {
  return JUST(PyUnpackGenerator(object_));
}

template<>
Maybe<one::Generator> PythonArg::ObjectAs<one::Generator>() const {
  return PyUnpackGenerator(object_);
}

template<>
Maybe<Symbol<Device>> PythonArg::ObjectAs<Symbol<Device>>() const {
  if (PyStringCheck(object_)) {
    std::string device_str = *JUST(PyStringAsString(object_));
    return Device::ParseAndNew(device_str);
  }
  return PyUnpackDevice(object_);
}

template<>
Maybe<Symbol<ParallelDesc>> PythonArg::ObjectAs<Symbol<ParallelDesc>>() const {
  return PyUnpackParallelDesc(object_);
}

template<>
Maybe<Symbol<cfg::SbpParallel>> PythonArg::ObjectAs<Symbol<cfg::SbpParallel>>() const {
  return PyUnpackSbpParallel(object_);
}

template<>
Maybe<std::vector<Symbol<cfg::SbpParallel>>>
PythonArg::ObjectAs<std::vector<Symbol<cfg::SbpParallel>>>() const {
  if (PySbpParallelCheck(object_)) {
    return std::make_shared<std::vector<Symbol<cfg::SbpParallel>>>(
        1, JUST(PyUnpackSbpParallel(object_)));
  }
  return PyUnpackSbpParallelSequence(object_);
}

template<>
Maybe<TensorIndex> PythonArg::ObjectAs<TensorIndex>() const {
  return PyUnpackTensorIndex(object_);
}

template<>
Maybe<std::shared_ptr<one::OpExpr>> PythonArg::ObjectAs<std::shared_ptr<one::OpExpr>>() const {
  return JUST(PyUnpackOpExpr(object_));
}

template<>
Maybe<one::OpExpr> PythonArg::ObjectAs<one::OpExpr>() const {
  return PyUnpackOpExpr(object_);
}

template<>
Maybe<PyObject*> PythonArg::ObjectAs<PyObject*>() const {
  return object_;
}

template<>
Maybe<const PyObject*> PythonArg::ObjectAs<const PyObject*>() const {
  return object_;
}

template<>
Maybe<std::vector<std::string>> PythonArg::ObjectAs<std::vector<std::string>>() const {
  return PyUnpackSequence<std::string>(
      object_, [](PyObject* item) -> Maybe<std::string> { return JUST(PyStringAsString(item)); });
}

Maybe<bool> PythonArg::TypeCheck(ValueType type) const {
  if (active_tag_ == HAS_IMMEDIATE) { return immediate_->value_type() == type; }
  switch (type) {
    case kINT32:
    case kUINT32:
    case kINT64:
    case kUINT64:
    case kBOOL: return PyLong_Check(object_);
    case kINT32_LIST:
    case kUINT32_LIST:
    case kINT64_LIST:
    case kUINT64_LIST:
    case kBOOL_LIST: return PyLongSequenceCheck(object_) || (size_ > 0 && PyLong_Check(object_));
    case kFLOAT:
    case kDOUBLE: return PyFloat_Check(object_) || PyLong_Check(object_);
    case kFLOAT_LIST:
    case kDOUBLE_LIST:
      return PyFloatSquenceCheck(object_)
             || (size_ > 0 && (PyFloat_Check(object_) || PyLong_Check(object_)));
    case kSTRING: return PyStringCheck(object_);
    case kSTRING_LIST: return PyStringSequenceCheck(object_);
    case kSCALAR: return PyScalarCheck(object_);
    case kTENSOR:
    case kTENSOR_REF: return PyTensorCheck(object_);
    case kTENSOR_TUPLE: return PyTensorTupleCheck(object_) || PyTensorSequenceCheck(object_);
    case kDTYPE: return PyDTypeCheck(object_);
    case kSHAPE: return PyLongSequenceCheck(object_);
    case kGENERATOR:
    case kGENERATOR_REF: return PyGeneratorCheck(object_);
    case kTENSOR_INDEX: return PyTensorIndexCheck(object_);
    case kDEVICE: return PyDeviceCheck(object_) || PyStringCheck(object_);
    case kPARALLEL_DESC: return PyParallelDescCheck(object_);
    case kSBP_PARALLEL: return PySbpParallelCheck(object_);
    case kSBP_PARALLEL_LIST:
      return PySbpParallelSequenceCheck(object_) || PySbpParallelCheck(object_);
    case kOPEXPR_REF: return PyOpExprCheck(object_);
    case kPY_OBJECT: return nullptr != object_;
    case kDTYPE_LIST: return PyDTypeSequenceCheck(object_);
    case kSHAPE_LIST: return PyShapeSequenceCheck(object_);
    default: {
      OF_UNIMPLEMENTED() << "Can not check type " << JUST(ValueTypeName(type));
    }
  }
  return false;
}

bool PythonArgCheck(const PythonArg& arg, ValueType type) {
  return arg.TypeCheck(type).GetOrThrow();
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
