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
Maybe<py::object> APINNGraphAdditionalVarNames(const std::shared_ptr<NNGraph>& graph) {
  const auto names = *JUST(graph->GetAdditionalVarOpNames());
  py::list name_list = py::cast(names);
  return py::cast<py::object>(name_list);
}
Maybe<py::object> APINNGraphAdditionalVarTensors(const std::shared_ptr<NNGraph>& graph) {
  const auto tensors = *JUST(graph->GetAdditionalVarOpTensors());
  py::list tensor_list = py::cast(tensors);
  return py::cast<py::object>(tensor_list);
}
}  // namespace

ONEFLOW_API_PYBIND11_MODULE("nn.graph.", m) {
  using namespace oneflow;
  py::class_<NNGraph, std::shared_ptr<NNGraph>>(m, "CNNGraph")
      .def(py::init<const std::string&>())
      .def_property_readonly("name", &NNGraph::job_name)
      .def("register_input_op_names_and_tensors", &NNGraph::RegisterInputOpNamesAndTensors)
      .def("register_output_op_names_and_tensors", &NNGraph::RegisterOutputOpNamesAndTensors)
      .def("register_variable_op_names_and_tensors", &NNGraph::RegisterVariableOpNamesAndTensors)
      .def("register_additional_variable_names_and_tensors",
           &NNGraph::RegisterAdditionalVarOpNamesAndTensorsToBeLoaded)
      .def_property_readonly("additional_var_names", &APINNGraphAdditionalVarNames)
      .def_property_readonly("additional_var_tensors", &APINNGraphAdditionalVarTensors)
      .def("complie_and_init_runtime", &NNGraph::CompileAndInitRuntime);

  m.def("RunLazyNNGraph", &RunLazyNNGraph);
  m.def("SoftSyncNNGraphBuffers", &SoftSyncNNGraphBuffers);
  m.def("AddTensorAsGraphLoss", &AddTensorAsGraphLoss);
  m.def("SaveJobToIR",
        [](const std::string& serialized_job, const std::string& path) -> Maybe<void> {
          Job job;
          CHECK_OR_RETURN(TxtString2PbMessage(serialized_job, &job));
          return SaveJobToIR(&job, path);
          ;
        });
  m.def("LoadSerializedJobFromIR", [](const std::string& path) -> Maybe<py::bytes> {
    Job job;
    JUST(LoadJobFromIR(&job, path));
    return py::bytes(job.SerializeAsString());
  });
}

}  // namespace oneflow
