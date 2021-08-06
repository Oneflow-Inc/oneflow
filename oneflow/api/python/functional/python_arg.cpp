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
#include "oneflow/core/common/data_type.cfg.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/functional/tensor_index.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

#define INSTANCE_CAST_OBJECT_AS(T)                                    \
  template<>                                                          \
  Maybe<T> PythonArg::ObjectAs<T>() const {                           \
    return detail::cast<T>(Borrow());                                 \
  }                                                                   \
  template<>                                                          \
  Maybe<std::vector<T>> PythonArg::ObjectAs<std::vector<T>>() const { \
    return detail::cast<std::vector<T>>(Borrow());                    \
  }

OF_PP_FOR_EACH_TUPLE(INSTANCE_CAST_OBJECT_AS,
                     ARITHMETIC_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(std::string));

#undef INSTANCE_CAST_OBJECT_AS

template<>
Maybe<Scalar> PythonArg::ObjectAs<Scalar>() const {
  py::object obj = Borrow();
  if (detail::isinstance<int32_t>(obj)) {
    return Scalar(JUST(detail::cast<int32_t>(obj)));
  } else if (detail::isinstance<int64_t>(obj)) {
    return Scalar(JUST(detail::cast<int64_t>(obj)));
  } else if (detail::isinstance<float>(obj)) {
    return Scalar(JUST(detail::cast<float>(obj)));
  } else if (detail::isinstance<double>(obj)) {
    return Scalar(JUST(detail::cast<double>(obj)));
  } else if (detail::isinstance<bool>(obj)) {
    return Scalar(JUST(detail::cast<bool>(obj)));
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "Can not convert to scalar from python object whose type is "
                                << *JUST(detail::cast<std::string>(py::str(py::type::of(obj))));
  }
}

template<>
Maybe<std::shared_ptr<one::Tensor>> PythonArg::ObjectAs<std::shared_ptr<one::Tensor>>() const {
  return detail::cast<std::shared_ptr<one::Tensor>>(Borrow());
}

template<>
Maybe<one::Tensor> PythonArg::ObjectAs<one::Tensor>() const {
  return *JUST(detail::cast<std::shared_ptr<one::Tensor>>(Borrow()));
}

template<>
Maybe<std::shared_ptr<one::TensorTuple>> PythonArg::ObjectAs<std::shared_ptr<one::TensorTuple>>()
    const {
  py::object obj = Borrow();
  if (detail::isinstance<one::TensorTuple>(obj)) {
    return detail::cast<std::shared_ptr<one::TensorTuple>>(obj);
  }

  const auto& v = JUST(detail::cast<std::vector<std::shared_ptr<one::Tensor>>>(obj));
  auto values = std::make_shared<one::TensorTuple>(v->size());
  for (int i = 0; i < v->size(); ++i) { values->at(i) = v->at(i); }
  return values;
}

template<>
Maybe<one::TensorTuple> PythonArg::ObjectAs<one::TensorTuple>() const {
  return *JUST(ObjectAs<std::shared_ptr<one::TensorTuple>>());
}

template<>
Maybe<std::shared_ptr<cfg::AttrValue>> PythonArg::ObjectAs<std::shared_ptr<cfg::AttrValue>>()
    const {
  py::object obj = Borrow();
  if (detail::isinstance<cfg::AttrValue>(obj)) {
    return detail::cast<std::shared_ptr<cfg::AttrValue>>(obj);
  }
  auto attr_value = std::make_shared<cfg::AttrValue>();
  if (detail::isinstance<int32_t>(obj)) {
    attr_value->set_at_int32(JUST(detail::cast<int32_t>(obj)));
  } else if (detail::isinstance<double>(obj)) {
    attr_value->set_at_double(JUST(detail::cast<double>(obj)));
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "The attribute type was not supported which is "
                                << *JUST(detail::cast<std::string>(py::str(py::type::of(obj))));
  }
  return attr_value;
}

template<>
Maybe<AttrMap> PythonArg::ObjectAs<AttrMap>() const {
  const auto& attrs = *(JUST(detail::cast<std::shared_ptr<MutableCfgAttrMap>>(Borrow())));
  return std::make_shared<AttrMap>(*attrs);
}

template<>
Maybe<DataType> PythonArg::ObjectAs<DataType>() const {
  py::object obj = Borrow();
  if (detail::isinstance<cfg::DataType>(obj)) {
    const auto& dtype = *JUST(detail::cast<std::shared_ptr<cfg::DataType>>(obj));
    return static_cast<DataType>(*dtype);
  } else if (detail::isinstance<DType>(obj)) {
    return JUST(detail::cast<DType&>(obj)).data_type();
  } else if (detail::isinstance<int32_t>(obj)) {
    return static_cast<DataType>(JUST(detail::cast<int32_t>(obj)));
  } else if (detail::isinstance<int64_t>(obj)) {
    return static_cast<DataType>(JUST(detail::cast<int64_t>(obj)));
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "Can not convert object to DataType from "
                                << *JUST(detail::cast<std::string>(py::str(py::type::of(obj))));
  }
  return kInvalidDataType;
}

template<>
Maybe<Shape> PythonArg::ObjectAs<Shape>() const {
  py::object obj = Borrow();
  if (detail::isinstance<Shape>(obj)) {
    return *JUST(detail::cast<std::shared_ptr<Shape>>(obj));
  } else if (detail::isinstance<py::list>(obj) || detail::isinstance<py::tuple>(obj)) {
    const auto& shape = JUST(ObjectAs<std::vector<int64_t>>());
    DimVector dim_vec(shape->size());
    for (int i = 0; i < shape->size(); ++i) { dim_vec[i] = shape->at(i); }
    return std::make_shared<Shape>(std::move(dim_vec));
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "Can not convert object to Shape from "
                                << *JUST(detail::cast<std::string>(py::str(py::type::of(obj))));
  }
}

template<>
Maybe<std::shared_ptr<one::Generator>> PythonArg::ObjectAs<std::shared_ptr<one::Generator>>()
    const {
  return detail::cast<std::shared_ptr<one::Generator>>(Borrow());
}

template<>
Maybe<one::Generator> PythonArg::ObjectAs<one::Generator>() const {
  return *JUST(detail::cast<std::shared_ptr<one::Generator>>(Borrow()));
}

template<>
Maybe<Symbol<Device>> PythonArg::ObjectAs<Symbol<Device>>() const {
  return **JUST(detail::cast<std::shared_ptr<Symbol<Device>>>(Borrow()));
}

template<>
Maybe<Symbol<ParallelDesc>> PythonArg::ObjectAs<Symbol<ParallelDesc>>() const {
  return **JUST(detail::cast<std::shared_ptr<Symbol<ParallelDesc>>>(Borrow()));
}

template<>
Maybe<Symbol<cfg::SbpParallel>> PythonArg::ObjectAs<Symbol<cfg::SbpParallel>>() const {
  return **JUST(detail::cast<std::shared_ptr<Symbol<cfg::SbpParallel>>>(Borrow()));
}

template<>
Maybe<std::vector<Symbol<cfg::SbpParallel>>>
PythonArg::ObjectAs<std::vector<Symbol<cfg::SbpParallel>>>() const {
  const auto& v =
      JUST(detail::cast<std::vector<std::shared_ptr<Symbol<cfg::SbpParallel>>>>(Borrow()));
  auto sbp_list = std::make_shared<std::vector<Symbol<cfg::SbpParallel>>>(v->size());
  for (int i = 0; i < v->size(); ++i) { sbp_list->at(i) = *(v->at(i)); }
  return sbp_list;
}

template<>
Maybe<TensorIndex> PythonArg::ObjectAs<TensorIndex>() const {
  auto tensor_index = std::make_shared<TensorIndex>();
  // Obvious single-entry cases.
  if (PySlice_Check(object_)         // NOLINT
      || PyLong_Check(object_)       // NOLINT
      || object_ == Py_Ellipsis      // NOLINT
      || object_ == Py_None          // NOLINT
      || PyTensorCheck(object_)      // NOLINT
      || !PySequence_Check(object_)  // NOLINT
      || PyUnicode_Check(object_)) {
    tensor_index->emplace_back(*JUST(detail::UnpackIndexItem(object_)));
    return tensor_index;
  }
  PyObject* tup = NULL;
  Py_ssize_t n = 0;
  if (PyTuple_Check(object_)) {
    tup = PySequence_Tuple(object_);
    n = PySequence_Size(tup);
  } else {
    // The follow comments are from numpy:
    // https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/mapping.c#L266
    /*
     * At this point, we're left with a non-tuple, non-array, sequence:
     * typically, a list. We use some somewhat-arbitrary heuristics from here
     * onwards to decided whether to treat that list as a single index, or a
     * list of indices.
     */
    n = PySequence_Size(object_);
    // Negative size indicates a Python error in the PySequence_Size call.
    if (n < 0) {
      PyErr_Clear();
      tensor_index->emplace_back(*JUST(detail::UnpackIndexItem(object_)));
      return tensor_index;
    }
    // The follow comments are from numpy:
    // https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/mapping.c#L280
    /*
     * Backwards compatibility only takes effect for short sequences - otherwise
     * we treat it like any other scalar.
     *
     * Sequences < NPY_MAXDIMS with any slice objects
     * or newaxis, Ellipsis or other arrays or sequences
     * embedded, are considered equivalent to an indexing
     * tuple. (`a[[[1,2], [3,4]]] == a[[1,2], [3,4]]`)
     */
    if (n >= /*NPY_MAXDIMS=*/32) {
      tensor_index->emplace_back(*JUST(detail::UnpackIndexItem(object_)));
      return tensor_index;
    }
    // Check whether we should unpack the index like a tuple.
    bool commit_to_unpack = false;
    for (Py_ssize_t i = 0; i < n; ++i) {
      PyObject* item = PySequence_GetItem(object_, i);
      if (commit_to_unpack) {
        CHECK_OR_RETURN(item) << "Sequence index is required.";
      } else {
        if (!item) {
          PyErr_Clear();
          break;
        }
        if (PySequence_Check(item)  // NOLINT
            || PySlice_Check(item)  // NOLINT
            || PyTensorCheck(item)  // NOLINT
            || item == Py_Ellipsis || item == Py_None) {
          commit_to_unpack = true;
        }
      }
      Py_DECREF(item);
    }
    if (commit_to_unpack) {
      tup = PySequence_Tuple(object_);
    } else {
      tensor_index->emplace_back(*JUST(detail::UnpackIndexItem(object_)));
      return tensor_index;
    }
  }

  tensor_index->resize(n);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_GetItem(tup, i);
    tensor_index->at(i) = *JUST(detail::UnpackIndexItem(item));
    Py_DECREF(item);
  }
  Py_DECREF(tup);
  return tensor_index;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
