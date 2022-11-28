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

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/autograd/autograd_function.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace py = pybind11;

namespace oneflow {

namespace {

// Transform input to TensorTuple
Maybe<one::TensorTuple> UnpackTensorTuple(const py::object& input) {
  one::TensorTuple tp;
  if (one::PyTensor_Check(input.ptr())) {
    tp.emplace_back(input.cast<std::shared_ptr<one::Tensor>>());
  } else if (py::isinstance<py::tuple>(input)) {
    auto tuple = input.cast<py::tuple>();
    tp.resize(tuple.size());
    for (int i = 0; i < tuple.size(); ++i) {
      PyObject* obj = tuple[i].ptr();
      if (obj == Py_None) {
        // do nothing
      } else if (one::PyTensor_Check(obj)) {
        tp[i] = one::PyTensor_Unpack(obj);
      } else {
        return Error::RuntimeError()
               << "expected Tensor or None as element " << i << ", but got "
               << one::functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(obj)));
      }
    }
  } else {
    return Error::RuntimeError()
           << "autograd.Function's output only support tensor or list of tensors";
  }
  return tp;
}

// Return single Tensor when TensorTuple's size is one, otherwise py::tuple
py::object PackTensorTuple(const one::TensorTuple& tp) {
  if (tp.size() == 1) {
    return py::cast(tp.at(0));
  } else {
    py::tuple out = py::tuple(tp.size());
    for (int i = 0; i < tp.size(); ++i) { out[i] = tp.at(i); }
    return py::cast<py::object>(out);
  }
}

// wrap PyFunction, unpack the inputs from TensorTuple and pack outputs to TensorTuple
one::AutogradFunctionBase::FType PackPyFunctionToFType(const py::function& func) {
  return [func](const std::shared_ptr<one::FunctionAutoGradCaptureState>& ctx,
                const one::TensorTuple& inputs) {
    const py::tuple& a = py::cast(inputs);
    py::object res = func(ctx, *a);
    return UnpackTensorTuple(res).GetPtrOrThrow();
  };
}

}  // namespace

namespace one {

ONEFLOW_API_PYBIND11_MODULE("autograd", m) {
  py::class_<AutogradFunctionBase, std::shared_ptr<AutogradFunctionBase>>(m, "AutogradFunctionBase")
      .def(py::init([]() { return std::make_shared<AutogradFunctionBase>(); }))
      .def_static("apply",
                  [](const std::string& name, const py::function& forward_fn,
                     const py::function& backward_fn, const py::args& input) -> Maybe<py::object> {
                    const auto& input_tensor_tuple = JUST(UnpackTensorTuple(input));
                    const std::shared_ptr<TensorTuple>& res = JUST(AutogradFunctionBase::Apply(
                        name, PackPyFunctionToFType(forward_fn), PackPyFunctionToFType(backward_fn),
                        *input_tensor_tuple));
                    return PackTensorTuple(*res);
                  });
}

}  // namespace one
}  // namespace oneflow
