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
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_ir.h"

namespace py = pybind11;

namespace oneflow {
namespace {
py::object APINNGraphAdditionalVarNames(const std::shared_ptr<NNGraph>& graph) {
  const auto names = *graph->GetAdditionalVarOpNames().GetPtrOrThrow();
  py::list name_list = py::cast(names);
  return py::cast<py::object>(name_list);
}
py::object APINNGraphAdditionalVarTensors(const std::shared_ptr<NNGraph>& graph) {
  const auto tensors = *graph->GetAdditionalVarOpTensors().GetPtrOrThrow();
  py::list tensor_list = py::cast(tensors);
  return py::cast<py::object>(tensor_list);
}
}  // namespace

ONEFLOW_API_PYBIND11_MODULE("nn.graph.", m) {
  using namespace oneflow;
  py::class_<NNGraph, std::shared_ptr<NNGraph>>(m, "CNNGraph")
      .def(py::init<const std::string&>())
      .def_property_readonly("name", &NNGraph::job_name)
      .def(
          "register_input_op_names_and_tensors",
          [](NNGraph& graph, const std::vector<std::string>& input_op_names,
             const std::vector<std::shared_ptr<one::Tensor>>& input_tensors) {
            return graph.RegisterInputOpNamesAndTensors(input_op_names, input_tensors).GetOrThrow();
          })
      .def("register_output_op_names_and_tensors",
           [](NNGraph& graph, const std::vector<std::string>& output_op_names,
              const std::vector<std::shared_ptr<one::Tensor>>& output_tensors) {
             return graph.RegisterOutputOpNamesAndTensors(output_op_names, output_tensors)
                 .GetOrThrow();
           })
      .def("register_variable_op_names_and_tensors",
           [](NNGraph& graph, const std::vector<std::string>& variable_op_names,
              const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors) {
             return graph.RegisterVariableOpNamesAndTensors(variable_op_names, variable_tensors)
                 .GetOrThrow();
           })
      .def("register_additional_variable_names_and_tensors",
           [](NNGraph& graph, const std::vector<std::string>& variable_op_names,
              const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors) {
             return graph
                 .RegisterAdditionalVarOpNamesAndTensorsToBeLoaded(variable_op_names,
                                                                   variable_tensors)
                 .GetOrThrow();
           })
      .def_property_readonly("additional_var_names", &APINNGraphAdditionalVarNames)
      .def_property_readonly("additional_var_tensors", &APINNGraphAdditionalVarTensors)
      .def("complie_and_init_runtime",
           [](NNGraph& graph) { return graph.CompileAndInitRuntime().GetOrThrow(); });

  m.def("RunLazyNNGraph",
        [](const one::TensorTuple& inputs, const one::TensorTuple& outputs,
           const one::TensorTuple& parameters, const std::shared_ptr<NNGraph>& nn_graph) {
          return RunLazyNNGraph(inputs, outputs, parameters, nn_graph).GetOrThrow();
        });
  m.def("SoftSyncNNGraphBuffers",
        [](const one::TensorTuple& buffers, const std::shared_ptr<NNGraph>& nn_graph) {
          return SoftSyncNNGraphBuffers(buffers, nn_graph).GetOrThrow();
        });
  m.def("AddTensorAsGraphLoss",
        [](const std::shared_ptr<one::Tensor>& t) { return AddTensorAsGraphLoss(t).GetOrThrow(); });
  m.def("SaveJobToIR", [](const std::string& serialized_job, const std::string& path) {
    Job job;
    CHECK(TxtString2PbMessage(serialized_job, &job));
    return SaveJobToIR(&job, path).GetOrThrow();
    ;
  });
  m.def("LoadSerializedJobFromIR", [](const std::string& path) {
    Job job;
    LoadJobFromIR(&job, path).GetOrThrow();
    return py::bytes(job.SerializeAsString());
  });
}

}  // namespace oneflow
