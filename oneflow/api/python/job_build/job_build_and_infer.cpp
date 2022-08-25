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
#include <string>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"

namespace py = pybind11;

namespace oneflow {

Maybe<void> MarkVariableGradients(const one::TensorTuple& variables,
                                  const one::TensorTuple& gradients) {
  CHECK_OR_RETURN(LazyMode::is_enabled());                 // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(variables.size(), gradients.size());  // NOLINT(maybe-need-error-msg)
  HashMap<std::string, std::string> variable_grad_lbns;
  for (int i = 0; i < variables.size(); ++i) {
    const std::string& variable_lbn = one::TensorNameScope::Global()->Lookup(variables[i]);
    CHECK_OR_RETURN(!variable_lbn.empty())
        << "variable which index is " << i << " expected to have a tensor name";
    const std::string& gradient_lbn = one::TensorNameScope::Global()->Lookup(gradients[i]);
    CHECK_OR_RETURN(!gradient_lbn.empty())
        << "gradient which index is " << i << " expected to have a tensor name";
    variable_grad_lbns.emplace(variable_lbn, gradient_lbn);
  }
  return JUST(GetCurInferCtx())->MarkVariableGradientBlobNames(variable_grad_lbns);
}

Maybe<void> MarkOutputGradients(const one::TensorTuple& outputs,
                                const one::TensorTuple& gradients) {
  CHECK_OR_RETURN(LazyMode::is_enabled());               // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), gradients.size());  // NOLINT(maybe-need-error-msg)
  HashMap<std::string, std::string> output_gradient_lbns;
  for (int i = 0; i < outputs.size(); ++i) {
    const std::string& output_lbn = one::TensorNameScope::Global()->Lookup(outputs[i]);
    CHECK_OR_RETURN(!output_lbn.empty())
        << "output which index is " << i << " expected to have a tensor name";
    const std::string& gradient_lbn = one::TensorNameScope::Global()->Lookup(gradients[i]);
    CHECK_OR_RETURN(!gradient_lbn.empty())
        << "gradient which index is " << i << " expected to have a tensor name";
    output_gradient_lbns.emplace(output_lbn, gradient_lbn);
  }
  return JUST(GetCurInferCtx())->MarkOutputGradientBlobNames(output_gradient_lbns);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("JobBuildAndInferCtx_Open", &JobBuildAndInferCtx_Open);
  m.def("JobBuildAndInferCtx_GetCurrentJobName", &JobBuildAndInferCtx_GetCurrentJobName);
  m.def("JobBuildAndInferCtx_GetCurrentJobId", &JobBuildAndInferCtx_GetCurrentJobId);
  m.def("JobBuildAndInferCtx_Close", &JobBuildAndInferCtx_Close);

  m.def("CurJobBuildAndInferCtx_SetJobConf", &CurJobBuildAndInferCtx_SetJobConf);

  m.def("CurJobBuildAndInferCtx_Complete", &CurJobBuildAndInferCtx_Complete,
        py::call_guard<py::gil_scoped_release>());
}

}  // namespace oneflow
