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

// wrap PyFunction, unpack the inputs from TensorTuple and pack outputs to TensorTuple
one::AutogradFunctionBase::FType PackPyFunctionToFType(const py::function& func) {
  return [&func](const std::shared_ptr<one::FunctionAutoGradCaptureState>& ctx,
                 const one::TensorTuple& inputs) {
    py::object res = func(ctx, py::args(py::make_tuple(inputs)));
    const auto& outputs = std::make_shared<one::TensorTuple>();
    if (py::isinstance<one::Tensor>(res)) {
      outputs->emplace_back(res.cast<std::shared_ptr<one::Tensor>>());
    } else if (py::isinstance<py::tuple>(res)) {
      for (const auto& tensor : res.cast<py::tuple>()) {
        CHECK(py::isinstance<one::Tensor>(tensor));
        outputs->emplace_back(tensor.cast<std::shared_ptr<one::Tensor>>());
      }
    } else {
      throw std::runtime_error("Forward and Backward function must return Tensor or TensorTuple.");
    }
    return outputs;
  };
}

}  // namespace

namespace one {

ONEFLOW_API_PYBIND11_MODULE("autograd", m) {
  py::class_<AutogradFunctionBase, std::shared_ptr<AutogradFunctionBase>>(m, "AutogradFunctionBase")
      .def(py::init([](const std::string& func_name, const py::function& forward_fn,
                       const py::function& backward_fn) {
        return std::make_shared<AutogradFunctionBase>(func_name, PackPyFunctionToFType(forward_fn),
                                                      PackPyFunctionToFType(backward_fn));
      }))
      .def("apply", [](const AutogradFunctionBase& func, const TensorTuple& inputs) {
        return func.Apply(inputs).GetPtrOrThrow();
      });

  py::class_<FunctionAutoGradCaptureState, std::shared_ptr<FunctionAutoGradCaptureState>>(
      m, "FunctionAutoGradCaptureState")
      .def(py::init([]() { return std::make_shared<FunctionAutoGradCaptureState>(); }))
      .def(
          "save_for_backward",
          [](const std::shared_ptr<FunctionAutoGradCaptureState>& ctx, const TensorTuple& tensors) {
            for (const auto& tensor : tensors) { ctx->SaveTensorForBackward(tensor); }
          })
      .def("saved_tensors",
           [](const FunctionAutoGradCaptureState& ctx) { return ctx.SavedTensors(); })
      .def("mark_non_differentiable", [](const std::shared_ptr<FunctionAutoGradCaptureState>& ctx,
                                         const TensorTuple& tensors) {
        for (const auto& tensor : tensors) { ctx->MarkNonDifferentiable(tensor); }
      });
}

}  // namespace one
}  // namespace oneflow
