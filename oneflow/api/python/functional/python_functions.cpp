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

py::object PyFmod(py::args py_args, py::kwargs py_kwargs) {
  // "broadcast_fmod(Tensor x, Tensor y)"
  // "scalar_fmod(Tensor in, Scalar scalar)"
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    CHECK_OR_RETURN(PyTensorCheck(input) || PyScalarCheck(input))
        << "The first input should be a tensor or scalar.";
    CHECK_OR_RETURN(PyTensorCheck(other) || PyScalarCheck(other))
        << "The second input should be a tensor or scalar.";
    if (PyTensorCheck(input) && PyTensorCheck(other)) {
      auto a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::BroadcastFMod(a, b);
    } else {
      if (PyTensorCheck(input)) {
        CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar.";
        auto a = JUST(PyUnpackTensor(input));
        auto b = *JUST(PyUnpackScalar(other));
        return functional::ScalarFMod(a, b);
      } else {
        CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar.";
        auto a = *JUST(PyUnpackScalar(input));
        auto b = JUST(PyUnpackTensor(other));
        return functional::ScalarFMod(b, a);
      }
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyEqual(py::args py_args, py::kwargs py_kwargs) {
  // "broadcast_equal(Tensor x, Tensor y)"
  // "scalar_logical_equal(Tensor in, Scalar scalar)"
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    CHECK_OR_RETURN(PyTensorCheck(input) || PyScalarCheck(input))
        << "The first input should be a tensor or scalar.";
    CHECK_OR_RETURN(PyTensorCheck(other) || PyScalarCheck(other))
        << "The second input should be a tensor or scalar.";

    if (PyTensorCheck(input) && PyTensorCheck(other)) {
      auto a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::BroadcastEqual(a, b);
    } else {
      if (PyTensorCheck(input)) {
        CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar.";
        auto a = JUST(PyUnpackTensor(input));
        auto b = *JUST(PyUnpackScalar(other));
        return functional::ScalarLogicalEqual(a, b);
      } else {
        CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar.";
        auto a = *JUST(PyUnpackScalar(input));
        auto b = JUST(PyUnpackTensor(other));
        return functional::ScalarLogicalEqual(b, a);
      }
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyNotEqual(py::args py_args, py::kwargs py_kwargs) {
  // "broadcast_not_equal(Tensor x, Tensor y)"
  // "scalar_logical_not_equal(Tensor in, Scalar scalar)"
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    CHECK_OR_RETURN(PyTensorCheck(input) || PyScalarCheck(input))
        << "The first input should be a tensor or scalar.";
    CHECK_OR_RETURN(PyTensorCheck(other) || PyScalarCheck(other))
        << "The second input should be a tensor or scalar.";

    if (PyTensorCheck(input) && PyTensorCheck(other)) {
      auto a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::BroadcastNotEqual(a, b);
    } else {
      if (PyTensorCheck(input)) {
        CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar.";
        auto a = JUST(PyUnpackTensor(input));
        auto b = *JUST(PyUnpackScalar(other));
        return functional::ScalarLogicalNotEqual(a, b);
      } else {
        CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar.";
        auto a = *JUST(PyUnpackScalar(input));
        auto b = JUST(PyUnpackTensor(other));
        return functional::ScalarLogicalNotEqual(b, a);
      }
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyGreater(py::args py_args, py::kwargs py_kwargs) {
  // "broadcast_greater(Tensor x, Tensor y)"
  // "scalar_logical_greater(Tensor in, Scalar scalar)"
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    CHECK_OR_RETURN(PyTensorCheck(input) || PyScalarCheck(input))
        << "The first input should be a tensor or scalar.";
    CHECK_OR_RETURN(PyTensorCheck(other) || PyScalarCheck(other))
        << "The second input should be a tensor or scalar.";

    if (PyTensorCheck(input) && PyTensorCheck(other)) {
      auto a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::BroadcastGreater(a, b);
    } else {
      if (PyTensorCheck(input)) {
        CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar.";
        auto a = JUST(PyUnpackTensor(input));
        auto b = *JUST(PyUnpackScalar(other));
        return functional::ScalarLogicalGreater(a, b);
      } else {
        CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar.";
        auto a = *JUST(PyUnpackScalar(input));
        auto b = JUST(PyUnpackTensor(other));
        return functional::ScalarLogicalGreater(b, a);
      }
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyGreaterEqual(py::args py_args, py::kwargs py_kwargs) {
  // "broadcast_greater_equal(Tensor x, Tensor y)"
  // "scalar_logical_greater_equal(Tensor in, Scalar scalar)"
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    CHECK_OR_RETURN(PyTensorCheck(input) || PyScalarCheck(input))
        << "The first input should be a tensor or scalar.";
    CHECK_OR_RETURN(PyTensorCheck(other) || PyScalarCheck(other))
        << "The second input should be a tensor or scalar.";

    if (PyTensorCheck(input) && PyTensorCheck(other)) {
      auto a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::BroadcastGreaterEqual(a, b);
    } else {
      if (PyTensorCheck(input)) {
        CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar.";
        auto a = JUST(PyUnpackTensor(input));
        auto b = *JUST(PyUnpackScalar(other));
        return functional::ScalarLogicalGreaterEqual(a, b);
      } else {
        CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar.";
        auto a = *JUST(PyUnpackScalar(input));
        auto b = JUST(PyUnpackTensor(other));
        return functional::ScalarLogicalGreaterEqual(b, a);
      }
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyLess(py::args py_args, py::kwargs py_kwargs) {
  // "broadcast_less(Tensor x, Tensor y)"
  // "scalar_logical_less(Tensor in, Scalar scalar)"
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    CHECK_OR_RETURN(PyTensorCheck(input) || PyScalarCheck(input))
        << "The first input should be a tensor or scalar.";
    CHECK_OR_RETURN(PyTensorCheck(other) || PyScalarCheck(other))
        << "The second input should be a tensor or scalar.";

    if (PyTensorCheck(input) && PyTensorCheck(other)) {
      auto a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::BroadcastLess(a, b);
    } else {
      if (PyTensorCheck(input)) {
        CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar.";
        auto a = JUST(PyUnpackTensor(input));
        auto b = *JUST(PyUnpackScalar(other));
        return functional::ScalarLogicalLess(a, b);
      } else {
        CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar.";
        auto a = *JUST(PyUnpackScalar(input));
        auto b = JUST(PyUnpackTensor(other));
        return functional::ScalarLogicalLess(b, a);
      }
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyLessEqual(py::args py_args, py::kwargs py_kwargs) {
  // "broadcast_less_equal(Tensor x, Tensor y)"
  // "scalar_logical_less_equal(Tensor in, Scalar scalar)"
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 2) << "2 positional inputs are required.";
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* input = PyTuple_GetItem(args, 0);
    PyObject* other = PyTuple_GetItem(args, 1);
    bool input_is_tensor = PyTensorCheck(input);
    bool other_is_tensor = PyTensorCheck(other);
    CHECK_OR_RETURN(input_is_tensor || other_is_tensor) << "Inputs must have one tensor at least.";
    CHECK_OR_RETURN(PyTensorCheck(input) || PyScalarCheck(input))
        << "The first input should be a tensor or scalar.";
    CHECK_OR_RETURN(PyTensorCheck(other) || PyScalarCheck(other))
        << "The second input should be a tensor or scalar.";

    if (PyTensorCheck(input) && PyTensorCheck(other)) {
      auto a = JUST(PyUnpackTensor(input));
      auto b = JUST(PyUnpackTensor(other));
      return functional::BroadcastLessEqual(a, b);
    } else {
      if (PyTensorCheck(input)) {
        CHECK_OR_RETURN(PyScalarCheck(other)) << "The second input should be a scalar.";
        auto a = JUST(PyUnpackTensor(input));
        auto b = *JUST(PyUnpackScalar(other));
        return functional::ScalarLogicalLessEqual(a, b);
      } else {
        CHECK_OR_RETURN(PyScalarCheck(input)) << "The first input should be a scalar.";
        auto a = *JUST(PyUnpackScalar(input));
        auto b = JUST(PyUnpackTensor(other));
        return functional::ScalarLogicalLessEqual(b, a);
      }
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

py::object PyWhere(py::args py_args, py::kwargs py_kwargs) {
  // "Where(Tensor condition, Tensor x, Tensor y)"
  // "WhereScalarX(Tensor condition, Scalar scalar, Tensor y)"
  // "WhereScalarY(Tensor condition, Tensor x, Scalar scalar)"
  // "WhereScalarXY(Tensor condition, Scalar x_scalar, Scalar y_scalar)"
  PyObject* args = py_args.ptr();
  size_t nargs = PyTuple_Size(args);
  CHECK_EQ_OR_THROW(nargs, 3) << "3 positional inputs are required.";
  const auto& result = [&]() -> Maybe<Tensor> {  // NOLINT
    PyObject* condition = PyTuple_GetItem(args, 0);
    PyObject* x = PyTuple_GetItem(args, 1);
    PyObject* y = PyTuple_GetItem(args, 2);

    CHECK_OR_RETURN(PyTensorCheck(condition)) << "condition should be a Tensor.";
    auto cond_tensor = JUST(PyUnpackTensor(condition));
    bool x_is_tensor = PyTensorCheck(x);
    bool y_is_tensor = PyTensorCheck(y);

    if (x_is_tensor && y_is_tensor) {
      auto x_tensor = JUST(PyUnpackTensor(x));
      auto y_tensor = JUST(PyUnpackTensor(y));
      return functional::Where(cond_tensor, x_tensor, y_tensor);
    } else if (!x_is_tensor && y_is_tensor) {
      CHECK_OR_RETURN(PyScalarCheck(x)) << "The x should be a scalar.";
      auto scalar = *JUST(PyUnpackScalar(x));
      auto y_tensor = JUST(PyUnpackTensor(y));
      return functional::WhereScalarX(cond_tensor, scalar, y_tensor);
    } else if (x_is_tensor && !y_is_tensor) {
      CHECK_OR_RETURN(PyScalarCheck(y)) << "The y should be a scalar.";
      auto x_tensor = JUST(PyUnpackTensor(x));
      auto scalar = *JUST(PyUnpackScalar(y));
      return functional::WhereScalarY(cond_tensor, x_tensor, scalar);
    } else {
      CHECK_OR_RETURN(PyScalarCheck(x) && PyScalarCheck(y))
          << "The x and y should be both a scalar.";
      auto x_scalar = *JUST(PyUnpackScalar(x));
      auto y_scalar = *JUST(PyUnpackScalar(y));
      return functional::WhereScalarXY(cond_tensor, x_scalar, y_scalar);
    }
  }();
  return py::cast(result.GetPtrOrThrow());
}

}  // namespace functional
}  // namespace one

namespace functional = one::functional;

ONEFLOW_API_PYBIND11_MODULE("F", m) {
  m.def("pow", &functional::PyPow);
  m.def("clamp", &functional::PyClamp);
  m.def("scatter", &functional::PyScatter);
  m.def("fmod", &functional::PyFmod);
  m.def("equal", &functional::PyEqual);
  m.def("not_equal", &functional::PyNotEqual);
  m.def("greater", &functional::PyGreater);
  m.def("greater_equal", &functional::PyGreaterEqual);
  m.def("less", &functional::PyLess);
  m.def("less_equal", &functional::PyLessEqual);
  m.def("where", &functional::PyWhere);
}

}  // namespace oneflow
