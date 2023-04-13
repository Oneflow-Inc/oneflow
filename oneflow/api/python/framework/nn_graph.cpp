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
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_ir.h"
#include "oneflow/core/job/job_interpreter.h"

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

Maybe<py::bytes> APINNGraphGetCurrentSerializedJob(const std::shared_ptr<NNGraph>& graph) {
  const auto job = graph->job();
  return py::bytes(job.SerializeAsString());
}
}  // namespace

ONEFLOW_API_PYBIND11_MODULE("nn.graph.", m) {
  using namespace oneflow;
  py::class_<NNGraph, std::shared_ptr<NNGraph>>(m, "CNNGraph")
      .def(py::init([](const std::string& name, const std::string& serialized_job, int64_t job_id,
                       const std::shared_ptr<MultiClientSessionContext>& session_ctx) {
        Job job;
        if (!job.ParseFromString(serialized_job)) {
          PyErr_SetString(PyExc_TypeError, "The second argument is not a valid job");
        }
        return std::make_shared<NNGraph>(name, job, job_id, session_ctx);
      }))
      .def(py::init([](const std::string& name, const std::string& serialized_plan, int64_t job_id,
                       const std::shared_ptr<MultiClientSessionContext>& session_ctx,
                       bool init_from_plan) {
        if (!init_from_plan) {
          PyErr_SetString(
              PyExc_TypeError,
              "init_from_plan must be True when init CNNGraph with this bool parameter.");
        }
        Plan plan;
        if (!plan.ParseFromString(serialized_plan)) {
          PyErr_SetString(PyExc_TypeError, "The second argument is not a valid plan");
        }
        return std::make_shared<NNGraph>(name, plan, job_id, session_ctx);
      }))
      .def_property_readonly("name", &NNGraph::job_name)
      .def_property(
          "job", /*getter*/
          [](const NNGraph& nn_graph) { return py::bytes(nn_graph.job().SerializeAsString()); },
          /*setter*/
          [](NNGraph& nn_graph, const std::string& serialized_job) {
            Job job;
            if (!job.ParseFromString(serialized_job)) {
              PyErr_SetString(PyExc_TypeError, "the value is not a valid job");
            }
            nn_graph.restore_job(job);
          })
      .def_property("job_id", &NNGraph::job_id,
                    [](NNGraph& nn_graph, int64_t job_id) { nn_graph.restore_job_id(job_id); })
      .def_property(
          "plan", /*getter*/
          [](const NNGraph& nn_graph) { return py::bytes(nn_graph.plan().SerializeAsString()); },
          /*setter*/
          [](NNGraph& nn_graph, const std::string& serialized_plan) {
            Plan plan;
            if (!plan.ParseFromString(serialized_plan)) {
              PyErr_SetString(PyExc_TypeError, "the value is not a valid plan");
            }
            nn_graph.restore_plan(plan);
          })
      .def("register_input_op_names_and_tensors", &NNGraph::RegisterInputOpNamesAndTensors)
      .def("register_output_op_names_and_tensors", &NNGraph::RegisterOutputOpNamesAndTensors)
      .def("register_variable_op_names_and_tensors", &NNGraph::RegisterVariableOpNamesAndTensors)
      .def("register_additional_variable_names_and_tensors",
           &NNGraph::RegisterAdditionalVarOpNamesAndTensorsToBeLoaded)
      .def_property_readonly("additional_var_names", &APINNGraphAdditionalVarNames)
      .def_property_readonly("additional_var_tensors", &APINNGraphAdditionalVarTensors)
      .def("align_states_after_logical_graph_compile",
           &NNGraph::AlignStatesAfterLogicalGraphCompile)
      .def("complete_graph_for_runtime", &NNGraph::CompleteLogicalGraphForRuntime)
      .def("build_with_new_input_from_shared_graph", &NNGraph::BuildWithNewInputFromSharedGraph)
      .def("compile_plan_for_runtime", &NNGraph::CompilePlanForRuntime)
      .def("init_runtime", &NNGraph::InitRuntime)
      .def("get_current_job_str", &APINNGraphGetCurrentSerializedJob);

  m.def("RunLazyNNGraph", &RunLazyNNGraph);
  m.def("RunLazyNNGraphByVM", &one::InterpretJob);
  m.def("SoftSyncNNGraphBuffers", &SoftSyncNNGraphBuffers);
  m.def("AddTensorAsGraphLoss", &AddTensorAsGraphLoss);
  m.def("MarkVariableGradients", [](const std::vector<std::shared_ptr<one::Tensor>>& variables,
                                    const std::vector<std::shared_ptr<one::Tensor>>& gradients) {
    one::TensorTuple variable_tuple(variables.size());
    one::TensorTuple gradient_tuple(gradients.size());
    for (int i = 0; i < variables.size(); ++i) { variable_tuple[i] = variables[i]; }
    for (int i = 0; i < gradients.size(); ++i) { gradient_tuple[i] = gradients[i]; }
    return MarkVariableGradients(variable_tuple, gradient_tuple);
  });
  m.def("ConvertJobToTosaIR", [](const std::string& serialized_job) -> Maybe<std::string> {
    Job job;
    CHECK_OR_RETURN(job.ParseFromString(serialized_job)) << "serialized job conversion failed.";
    return ConvertJobToTosaIR(&job);
  });
  m.def(
      "SaveJobToIR", [](const std::string& serialized_job, const std::string& path) -> Maybe<void> {
        Job job;
        CHECK_OR_RETURN(job.ParseFromString(serialized_job)) << "serialized job conversion failed.";
        return SaveJobToIR(&job, path);
      });
  m.def("ConvertJobToIR", [](const std::string& serialized_job) -> Maybe<std::string> {
    Job job;
    CHECK_OR_RETURN(job.ParseFromString(serialized_job)) << "serialized job conversion failed.";
    return ConvertJobToIR(&job);
  });
  m.def("LoadSerializedJobFromIR", [](const std::string& path) -> Maybe<py::bytes> {
    Job job;
    JUST(LoadJobFromIR(&job, path));
    return py::bytes(job.SerializeAsString());
  });
}

}  // namespace oneflow
