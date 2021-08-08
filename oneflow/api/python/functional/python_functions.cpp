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
#include <pybind11/pybind11.h>

#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/function_def.h"
#include "oneflow/api/python/functional/py_function.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

py::object PyAdd(py::args py_args, py::kwargs py_kwargs) {
  // "Add(Tensor input, Tensor other, *, Scalar alpha, bool inplace=False)"
  // "Add(Tensor input, Scalar other, *, Scalar alpha, bool inplace=False)"
  // "Add(Scalar input, Tensor other, *, Scalar alpha, bool inplace=False)"
  PyObject* args = py_args.ptr();
  PyObject* kwargs = py_kwargs.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  bool inplace = false;
  if (auto* obj = PyDict_GetItemString(kwargs, "inplace")) {
    CHECK_OR_THROW(PyBool_Check(obj)) << "The keyword inplace's value should be boolean.";
    inplace = (obj == Py_True);
  }
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    Optional<Scalar> alpha;
    if (auto* obj = PyDict_GetItemString(kwargs, "alpha")) { alpha = *JUST(PyUnpackScalar(obj)); }
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    CHECK_OR_RETURN(input_is_tensor || PyTensorCheck(other))
        << "Inputs must have one tensor at least.";
    if (!input_is_tensor) {
      CHECK_OR_RETURN(!inplace) << "Can not apply inplace on scalar input.";
      input = other;
      other = PyTuple_GetItem(args, 0);
    }
    auto a = JUST(PyUnpackTensor(input));

    if (PyScalarCheck(other)) {
      auto b = *JUST(PyUnpackScalar(other));
      if (alpha) {
        if (!input_is_tensor) {
          a = JUST(functional::ScalarMul(a, *JUST(alpha.value())));
        } else {
          b *= *JUST(alpha.value());
        }
      }
      return functional::ScalarAdd(a, b, inplace);
    } else {
      CHECK_OR_RETURN(PyTensorCheck(other)) << "The second input should be a scalar or tensor.";
      auto b = JUST(PyUnpackTensor(other));
      if (alpha) {
        if (!input_is_tensor) {
          a = JUST(functional::ScalarMul(a, *JUST(alpha.value())));
        } else {
          b = JUST(functional::ScalarMul(b, *JUST(alpha.value())));
        }
      }
      /* if (a->shape()->NumAxes() == 0) {
        return functional::ScalarAddByTensor(b, a, inplace);
      } else if (b->shape()->NumAxes() == 0) {
        return functional::ScalarAddByTensor(a, b, inplace);
      } else */
      if (*a->shape() == *b->shape()) {
        return functional::Add(a, b, inplace);
      } else {
        if (inplace) {
          const auto& tmp = functional::Expand(b, *a->shape());
          CHECK_OR_RETURN(tmp.IsOk()) << "Can not apply inplace on the broadcasting input.";
          return functional::Add(a, JUST(tmp), inplace);
        }
        return functional::BroadcastAdd(a, b);
      }
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PySub(py::args py_args, py::kwargs py_kwargs) {
  // "Sub(Tensor input, Tensor other, *, Scalar alpha, bool inplace=False)"
  // "Sub(Tensor input, Scalar other, *, Scalar alpha, bool inplace=False)"
  // "Sub(Scalar input, Tensor other, *, Scalar alpha)"
  PyObject* args = py_args.ptr();
  PyObject* kwargs = py_kwargs.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  bool inplace = false;
  if (auto* obj = PyDict_GetItemString(kwargs, "inplace")) {
    CHECK_OR_THROW(PyBool_Check(obj)) << "The keyword inplace's value should be boolean.";
    inplace = (obj == Py_True);
  }
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    Optional<Scalar> alpha;
    if (auto* obj = PyDict_GetItemString(kwargs, "alpha")) { alpha = *JUST(PyUnpackScalar(obj)); }
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    if (input_is_tensor && other_is_tensor) {
      // "Sub(Tensor input, Tensor other, *, Scalar alpha, bool inplace=False)"
      const auto& a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      if (alpha) { b = JUST(functional::ScalarMul(a, *JUST(alpha.value()))); }
      CHECK_OR_RETURN(!inplace) << "Can not apply inplace for broadcasting sub.";
      return functional::BroadcastSub(a, b);
    } else if (input_is_tensor && !other_is_tensor) {
      // "Sub(Tensor input, Scalar other, *, Scalar alpha, bool inplace=False)"
      const auto& a = JUST(PyUnpackTensor(input));
      CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar or tensor.";
      auto b = *JUST(PyUnpackScalar(other));
      b *= -1;
      if (alpha) { b *= *JUST(alpha.value()); }
      return functional::ScalarAdd(a, b, inplace);
    } else {
      // "Sub(Scalar input, Tensor other, *, Scalar alpha)"
      CHECK_OR_RETURN(!inplace) << "Can not apply inplace on scalar input.";
      CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar or tensor.";
      Scalar a = *JUST(PyUnpackScalar(input));
      auto b = JUST(PyUnpackTensor(other));
      Scalar multiplier = -1;
      if (alpha) { multiplier *= *JUST(alpha.value()); }
      b = JUST(functional::ScalarMul(b, multiplier));
      return functional::ScalarAdd(b, a, /*inplace=*/false);
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyMul(py::args py_args, py::kwargs py_kwargs) {
  // "Mul(Tensor input, Tensor other, *, bool inplace=False)"
  // "Mul(Tensor input, Scalar other, *, bool inplace=False)"
  // "Mul(Scalar input, Tensor other)"
  PyObject* args = py_args.ptr();
  PyObject* kwargs = py_kwargs.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  bool inplace = false;
  if (auto* obj = PyDict_GetItemString(kwargs, "inplace")) {
    CHECK_OR_THROW(PyBool_Check(obj)) << "The keyword inplace's value should be boolean.";
    inplace = (obj == Py_True);
  }
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    CHECK_OR_RETURN(!inplace) << "Can not apply inplace for mul.";
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    if (input_is_tensor && other_is_tensor) {
      // "Mul(Tensor input, Tensor other, *, bool inplace=False)"
      const auto& a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      if (*a->shape() == *b->shape()) { return functional::Multiply(a, b); }
      return functional::BroadcastMul(a, b);
    } else if (input_is_tensor && !other_is_tensor) {
      // "Mul(Tensor input, Scalar other, *, bool inplace=False)"
      const auto& a = JUST(PyUnpackTensor(input));
      CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar or tensor.";
      auto b = *JUST(PyUnpackScalar(other));
      return functional::ScalarMul(a, b);
    } else {
      // "Mul(Scalar input, Tensor other)"
      CHECK_OR_RETURN(!inplace) << "Can not apply inplace on scalar input.";
      CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar or tensor.";
      Scalar a = *JUST(PyUnpackScalar(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::ScalarMul(b, a);
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyDiv(py::args py_args, py::kwargs py_kwargs) {
  // "Div(Tensor input, Tensor other, *, bool inplace=False)"
  // "Div(Tensor input, Scalar other, *, bool inplace=False)"
  // "Div(Scalar input, Tensor other)"
  PyObject* args = py_args.ptr();
  PyObject* kwargs = py_kwargs.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  bool inplace = false;
  if (auto* obj = PyDict_GetItemString(kwargs, "inplace")) {
    CHECK_OR_THROW(PyBool_Check(obj)) << "The keyword inplace's value should be boolean.";
    inplace = (obj == Py_True);
  }
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    CHECK_OR_RETURN(!inplace) << "Can not apply inplace for div.";
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    if (input_is_tensor && other_is_tensor) {
      // "Div(Tensor input, Tensor other, *, bool inplace=False)"
      const auto& a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::BroadcastDiv(a, b);
    } else if (input_is_tensor && !other_is_tensor) {
      // "Div(Tensor input, Scalar other, *, bool inplace=False)"
      const auto& a = JUST(PyUnpackTensor(input));
      CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar or tensor.";
      Scalar b = 1.0;
      b /= *JUST(PyUnpackScalar(other));
      return functional::ScalarMul(a, b);
    } else {
      // "Div(Scalar input, Tensor other)"
      CHECK_OR_RETURN(!inplace) << "Can not apply inplace on scalar input.";
      CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar or tensor.";
      Scalar a = *JUST(PyUnpackScalar(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::ScalarMul(JUST(functional::ReciprocalNoNan(b)), a);
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyPow(py::args py_args, py::kwargs py_kwargs) {
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";

  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* exponent = PyTuple_GetItem(args, 1);

    CHECK_OR_RETURN(PyTensorCheck(input)) << "input should be a Tensor.";
    const auto& in = JUST(PyUnpackTensor(input));

    bool is_exponent_scalar = PyScalarCheck(exponent);
    bool is_exponent_tensor = PyTensorCheck(exponent);

    CHECK_OR_RETURN(is_exponent_scalar || is_exponent_tensor)
        << "exponent should be a scalar or Tensor.";

    if (is_exponent_scalar) {
      Scalar exponent_scalar = *JUST(PyUnpackScalar(exponent));
      return functional::ScalarPow(in, exponent_scalar);
    } else if (is_exponent_tensor) {
      const auto& exp = JUST(PyUnpackTensor(exponent));
      CHECK_OR_RETURN(*(exp->shape()) == *(in->shape()))
          << "shape of input and exp should be same.";
      return functional::Pow(in, exp);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "pow unimplemented branch.";
    }
  }();

  return py::cast(result.GetPtrOrThrow());
}

py::object PyClamp(py::args py_args, py::kwargs py_kwargs) {
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 3) << "3 positional inputs are required.";

  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* min = PyTuple_GetItem(args, 1);
    PyObject* max = PyTuple_GetItem(args, 2);

    CHECK_OR_RETURN(PyTensorCheck(input)) << "input type should be a Tensor.";
    const auto& in = JUST(PyUnpackTensor(input));

    bool has_min_bound = (min != Py_None);
    bool has_max_bound = (max != Py_None);
    if (has_min_bound) { CHECK_OR_RETURN(PyScalarCheck(min)) << "min should be scalar or None."; }
    if (has_max_bound) { CHECK_OR_RETURN(PyScalarCheck(max)) << "max should be scalar or None."; }
    if (has_min_bound && has_max_bound) {
      Scalar min_value = *JUST(PyUnpackScalar(min));
      Scalar max_value = *JUST(PyUnpackScalar(max));
      return functional::ClipByScalar(in, min_value, max_value);
    } else if (has_min_bound && !has_max_bound) {
      Scalar min_value = *JUST(PyUnpackScalar(min));
      return functional::ClipByScalarMin(in, min_value);
    } else if (!has_min_bound && has_max_bound) {
      Scalar max_value = *JUST(PyUnpackScalar(max));
      return functional::ClipByScalarMax(in, max_value);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "min and max cannot be None at the same time.";
    }
  }();

  return py::cast(result.GetPtrOrThrow());
}

py::object PyScatter(py::args py_args, py::kwargs py_kwargs) {
  // Scatter(Tensor input, Int32 dim, Tensor index, Tensor src)
  // Scatter(Tensor input, Int32 dim, Tensor index, float src)

  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 4) << "4 positional inputs are required.";

  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* dim = PyTuple_GetItem(args, 1);
    PyObject* index = PyTuple_GetItem(args, 2);
    PyObject* src = PyTuple_GetItem(args, 3);

    CHECK_OR_RETURN(PyTensorCheck(input)) << "input should be a Tensor.";
    const auto& in = JUST(PyUnpackTensor(input));

    CHECK_OR_RETURN(PyScalarCheck(dim)) << "dim should be a scalar.";
    Scalar dim_scalar = *JUST(PyUnpackScalar(dim));
    int32_t d = JUST(dim_scalar.As<int32_t>());

    CHECK_OR_RETURN(PyTensorCheck(index)) << "index should be a Tensor.";
    const auto& idx = JUST(PyUnpackTensor(index));

    if (PyTensorCheck(src)) {
      const auto& src_tensor = JUST(PyUnpackTensor(src));
      return functional::DimScatter(in, idx, src_tensor, d);
    }
    CHECK_OR_RETURN(PyScalarCheck(src)) << "src should be a scalar or Tensor.";
    Scalar src_scalar = *JUST(PyUnpackScalar(src));
    return functional::DimScatterUpdateScalar(in, idx, JUST(src_scalar.As<float>()), d);
  }();
  return py::cast(result.GetPtrOrThrow());
}

}  // namespace functional
}  // namespace one

namespace functional = one::functional;

ONEFLOW_API_PYBIND11_MODULE("F", m) {
  m.def("add", &functional::PyAdd);
  m.def("sub", &functional::PySub);
  m.def("mul", &functional::PyMul);
  m.def("div", &functional::PyDiv);
  m.def("pow", &functional::PyPow);
  m.def("clamp", &functional::PyClamp);
  m.def("scatter", &functional::PyScatter);
}

}  // namespace oneflow
