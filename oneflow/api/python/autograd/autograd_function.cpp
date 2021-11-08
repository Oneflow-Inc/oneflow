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
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/autograd/autograd_function.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace py = pybind11;

namespace oneflow {

namespace {

// Transform input to TensorTuple
Maybe<one::TensorTuple> UnpackTensorTuple(const py::object& input) {
  one::TensorTuple tp;
  if (py::isinstance<one::Tensor>(input)) {
    tp.emplace_back(input.cast<std::shared_ptr<one::Tensor>>());
  } else if (py::isinstance<py::tuple>(input)) {
    for (const auto& tensor : input.cast<py::tuple>()) {
      CHECK_OR_RETURN(py::isinstance<one::Tensor>(tensor));
      tp.emplace_back(tensor.cast<std::shared_ptr<one::Tensor>>());
    }
  } else {
    throw std::runtime_error("Only support tensor or list of tensors");
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
      .def_static("apply", [](const std::string& name, const py::function& forward_fn,
                              const py::function& backward_fn, const py::args& input) {
        const auto& input_tensor_tuple = UnpackTensorTuple(input).GetOrThrow();
        const std::shared_ptr<TensorTuple>& res =
            AutogradFunctionBase::Apply(name, PackPyFunctionToFType(forward_fn),
                                        PackPyFunctionToFType(backward_fn), input_tensor_tuple)
                .GetPtrOrThrow();
        return PackTensorTuple(*res);
      });

  py::class_<FunctionAutoGradCaptureState, std::shared_ptr<FunctionAutoGradCaptureState>>(
      m, "FunctionAutoGradCaptureState")
      .def(py::init([]() { return std::make_shared<FunctionAutoGradCaptureState>(); }))
      .def("save_for_backward",
           [](FunctionAutoGradCaptureState& ctx, const py::args& input) {
             const auto& tensors = UnpackTensorTuple(input).GetOrThrow();
             for (const auto& tensor : tensors) { ctx.SaveTensorForBackward(tensor); }
           })
      .def_property_readonly(
          "saved_tensors",
          [](const FunctionAutoGradCaptureState& ctx) { return py::cast(ctx.SavedTensors()); })
      .def("mark_non_differentiable", [](FunctionAutoGradCaptureState& ctx, const py::args& input) {
        const auto& tensors = UnpackTensorTuple(input).GetOrThrow();
        for (const auto& tensor : tensors) { ctx.MarkNonDifferentiable(tensor); }
      });
}

}  // namespace one
}  // namespace oneflow
