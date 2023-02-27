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
#ifndef ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_H_
#define ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_H_

#include <memory>
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {

class Blob;

class NNGraph final : public NNGraphIf {
 public:
  explicit NNGraph(const std::string& name, const Job& job, int64_t job_id,
                   const std::shared_ptr<MultiClientSessionContext>& session_ctx)
      : name_(name),
        job_(job),
        job_id_(job_id),
        session_ctx_(session_ctx),
        runtime_inited_(false),
        is_closed_(false) {}
  explicit NNGraph(const std::string& name, const Plan& plan, int64_t job_id,
                   const std::shared_ptr<MultiClientSessionContext>& session_ctx)
      : name_(name),
        job_id_(job_id),
        session_ctx_(session_ctx),
        plan_(plan),
        runtime_inited_(false),
        is_closed_(false) {}
  OF_DISALLOW_COPY_AND_MOVE(NNGraph);
  ~NNGraph();

  const std::string& job_name() const override { return name_; }
  const Job& job() const { return job_; }
  void restore_job(const Job& job) { job_ = job; }
  int64_t job_id() const { return job_id_; }
  void restore_job_id(int64_t job_id) { job_id_ = job_id; }
  const Plan& plan() const { return plan_; }
  void restore_plan(const Plan& plan) { plan_ = plan; }
  const std::vector<std::string>& inputs_op_names() const override;
  const std::vector<std::string>& outputs_op_names() const override;
  const std::vector<bool>& inputs_valid() const override;
  const std::vector<bool>& outputs_valid() const override;
  const std::vector<std::string>& inputs_tensor_meta_str() const;
  const std::vector<std::string>& outputs_tensor_meta_str() const;
  int64_t variable_op_size() const;
  const std::shared_ptr<vm::EagerBlobObjectList>& var_blobs() const;

  Maybe<void> RegisterAdditionalVarOpNamesAndTensorsToBeLoaded(
      const std::vector<std::string>& additional_var_names,
      const std::vector<std::shared_ptr<one::Tensor>>& additional_var_tensors);
  Maybe<void> RegisterInputOpNamesAndTensors(
      const std::vector<std::string>& inputs_op_names,
      const std::vector<std::shared_ptr<one::Tensor>>& input_tensors);
  Maybe<void> RegisterOutputOpNamesAndTensors(
      const std::vector<std::string>& outputs_op_names,
      const std::vector<std::shared_ptr<one::Tensor>>& output_tensors);
  Maybe<void> RegisterVariableOpNamesAndTensors(
      const std::vector<std::string>& variable_op_names,
      const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors);
  Maybe<std::vector<std::string>> GetAdditionalVarOpNames() const;
  Maybe<std::vector<std::shared_ptr<one::Tensor>>> GetAdditionalVarOpTensors() const;
  // After logical graph compile, some state variables should be cleaned or built.
  Maybe<void> AlignStatesAfterLogicalGraphCompile();
  // Add special operators into logical graph for lazy runtime.
  Maybe<void> CompleteLogicalGraphForRuntime();
  // Build graph with new inputs from a completed job of a shared graph.
  Maybe<void> BuildWithNewInputFromSharedGraph(
      const std::vector<std::string>& shared_inputs_op_names,
      const std::vector<std::shared_ptr<one::Tensor>>& new_input_tensors,
      const std::vector<std::string>& shared_op_names_from_ordered_original_graph,
      const std::string& new_serialized_original_job);
  // Generate execution plan for lazy runtime. Oneflow lazy runtime is an actor based runtime.
  Maybe<void> CompilePlanForRuntime();
  // Initialize lazy runtime.
  Maybe<void> InitRuntime();
  Maybe<void> CompileAndInitRuntime();
  Maybe<void> Close();
  const auto variable_op_name2tensor() const { return variable_op_name2tensor_; }
  std::vector<std::shared_ptr<one::UserOpExpr>> cached_op_exprs;

 private:
  Maybe<void> RegisterFreeEagerTensorsToVariableOpNames();
  Maybe<void> RegisterNewVariableOpInJobPass();
  Maybe<void> DeleteOutdatedVariableInVariableTensorMgr();
  Maybe<void> GetVariableRealBlobAfterSyncPlan();

  void NewRuntimeBuffers();
  void CloseRuntimeBuffers();

  std::string name_;
  Job job_;
  int64_t job_id_;
  std::shared_ptr<MultiClientSessionContext> session_ctx_;
  std::vector<std::string> inputs_op_names_;
  std::vector<std::string> outputs_op_names_;
  std::vector<bool> input_tensors_valid_;
  std::vector<bool> output_tensors_valid_;
  std::vector<std::string> inputs_tensor_meta_str_;
  std::vector<std::string> outputs_tensor_meta_str_;
  HashMap<std::string, std::shared_ptr<one::Tensor>> variable_op_name2tensor_;
  // Additional variables are variable other than model states, such as states in
  // optimizers/lr schedulers or free eager tensors.
  HashSet<std::string> additional_variable_op_name_;
  // Additional states tensor loaded from state dict,
  // they will be load into job after plan is generated.
  HashMap<std::string, std::shared_ptr<one::Tensor>>
      additional_variable_op_tobe_loaded_name2tensor_;
  HashMap<std::string, vm::EagerBlobObject*> variable_op_name2eager_blob_object_;
  HashSet<std::string> variable_op_names_;
  std::shared_ptr<vm::EagerBlobObjectList> variable_op_blobs_;
  Plan plan_;
  // TODO(chengcheng): temp impl using runtime now, need reimplement for dynamic multi nn.Graph.
  std::unique_ptr<Runtime> runtime_;
  bool runtime_inited_;
  bool is_closed_;
};

Maybe<void> RunLazyNNGraph(const one::TensorTuple& inputs, const one::TensorTuple& outputs,
                           const std::shared_ptr<NNGraph>& nn_graph);

Maybe<void> SoftSyncNNGraphBuffers(const one::TensorTuple& buffers,
                                   const std::shared_ptr<NNGraph>& nn_graph);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_H_
