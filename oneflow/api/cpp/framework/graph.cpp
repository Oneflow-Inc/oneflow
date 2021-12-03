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

#include "oneflow/api/common/scope.h"
#include "oneflow/api/cpp/framework/device.h"
#include "oneflow/api/cpp/framework/graph.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/api/common/job_build_and_infer_ctx.h"
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/eager/eager_oneflow.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_ir.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/operator/interface_blob_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow_api {

namespace of = oneflow;

namespace {

class CompileScope {
 public:
  CompileScope(const of::JobConfigProto& job_config, const std::shared_ptr<of::NNGraph>& graph_) {
    of::Global<of::MultiClientSessionContext>::New();
    of::ConfigProto config_proto;
    config_proto.mutable_resource()->set_cpu_device_num(1);
    config_proto.set_session_id(of::NewSessionId());
    of::Global<of::MultiClientSessionContext>::Get()->TryInit(config_proto).GetOrThrow();
    of::Global<of::MultiClientSessionContext>::Get()->AddCGraph(graph_).GetOrThrow();

    std::shared_ptr<of::Scope> scope = of::MakeInitialScope(job_config).GetOrThrow();
    of::ThreadLocalScopeStackPush(scope).GetOrThrow();
    of::JobBuildAndInferCtx_Open(job_config.job_name()).GetOrThrow();
    of::cfg::JobConfigProto job_config_cfg(job_config);
    of::CurJobBuildAndInferCtx_SetJobConf(job_config_cfg).GetOrThrow();
  }

  ~CompileScope() {
    of::JobBuildAndInferCtx_Close().GetOrThrow();
    of::ThreadLocalScopeStackPop().GetOrThrow();
  }

 private:
  of::LazyMode::Guard lazy_mode_enabled_guard{true};
};

std::shared_ptr<of::one::TensorTuple> ConvertToTensorTuple(
    const std::vector<std::shared_ptr<of::one::Tensor>>& tensors) {
  auto tensor_tuple = std::make_shared<of::one::TensorTuple>();
  for (const auto& tensor : tensors) { tensor_tuple->emplace_back(tensor); }
  return tensor_tuple;
}

}  // namespace

Graph::Graph(const std::string& model_path, const Device& device) : device_(device) {
  // TODO(zzk0): model_path is a directory, need to concatenate filename
  // we need a mlir model name.
  // of::LoadJobFromIR(&job_, model_path).GetOrThrow();

  std::ifstream input("/home/zhouzekai/oneflow/oneflow/ir/test/saved_model/job.pb");
  std::stringstream buffer;
  buffer << input.rdbuf();
  job_.ParseFromString(buffer.str());
  input.close();

  graph_ = std::make_shared<of::NNGraph>(job_.job_conf().job_name());
}

Graph::Graph(const std::string& model_path) : Graph(model_path, Device("cpu")) {}

std::vector<Tensor> Graph::Forward(const std::vector<Tensor>& inputs) {
  if (!is_compiled_) {
    Compile(inputs);
    is_compiled_ = true;
  }
  return Run(inputs);
}

void Graph::SetBatchSize(int batch_size) { batch_size_ = batch_size; }

void Graph::Compile(const std::vector<Tensor>& inputs) {
  BuildGraph(inputs);
  LoadCheckpoint();
  RegisterTensors();
  graph_->CompileAndInitRuntime().GetOrThrow();
}

std::vector<Tensor> Graph::Run(const std::vector<Tensor>& inputs) {
  auto input_tensor_tuple = std::make_shared<of::one::TensorTuple>();
  for (const auto& tensor : inputs) { input_tensor_tuple->emplace_back(tensor.tensor_); }

  of::RunLazyNNGraph(*input_tensor_tuple, *output_tensor_tuple_, *parameter_tensor_tuple_, graph_)
      .GetOrThrow();
  of::SoftSyncNNGraphBuffers(*output_tensor_tuple_, graph_).GetOrThrow();

  std::vector<Tensor> outputs;
  for (const auto& tensor : *output_tensor_tuple_) { outputs.emplace_back(Tensor(tensor)); }
  return outputs;
}

void Graph::AddOp(oneflow::OperatorConf op_conf) {
  std::shared_ptr<of::Scope> scope = oneflow::GetCurrentScope().GetPtrOrThrow();
  op_conf.set_scope_symbol_id(scope->symbol_id().value_or(0));
  if (device_.type() == "cpu") {
    op_conf.set_device_tag(device_.type());
  } else {
    op_conf.set_device_tag(device_.type() + ":" + std::to_string(device_.device_id()));
  }
  if (batch_size_ > 0) {
    op_conf.mutable_input_conf()->mutable_blob_conf()->mutable_shape()->mutable_dim()->Set(
        0, batch_size_);
  }
  auto* ctx = of::GetCurInferCtx().GetOrThrow();
  ctx->AddAndInferConsistentOp(op_conf).GetOrThrow();
}

void Graph::BuildGraph(const std::vector<Tensor>& inputs) {
  CompileScope build_graph_scope(job_.job_conf(), graph_);

  // TODO(zzk0): remove this; used for input tensor order
  int input_tensor_order = 0;

  of::OpGraph op_graph(job_);
  op_graph
      .ForEachOpNode([&](const of::OpNode& node) -> of::Maybe<void> {
        const oneflow::OperatorConf& op_conf = node.op().op_conf();
        AddOp(op_conf);
        if (op_conf.has_input_conf()) {
          // TODO(zzk0): input tensor order
          input_name_to_tensor_[op_conf.name()] = inputs.at(input_tensor_order).tensor_;
          ++input_tensor_order;
        } else if (op_conf.has_variable_conf()) {
          // TODO(zzk0): load from local path
          of::LazyMode::Guard lazy_mode_disabled_guard{false};

          oneflow::VariableOpConf variable_conf = op_conf.variable_conf();
          variable_op_name_to_tensor_[op_conf.name()] =
              of::one::functional::Empty(
                  of::Shape(variable_conf.shape()),
                  of::DType::Get(static_cast<of::DataType>(variable_conf.data_type())).GetOrThrow(),
                  *device_.device_)
                  .GetPtrOrThrow();
        }
        return of::Maybe<void>::Ok();
      })
      .GetOrThrow();

  of::CurJobBuildAndInferCtx_Complete().GetOrThrow();
  of::CurJobBuildAndInferCtx_Rebuild().GetOrThrow();
  oneflow::Job complete_job = oneflow::GetCurrentJob().GetOrThrow();
  of::OpGraph complete_graph(complete_job);
  complete_graph
      .ForEachOpNode([&](const of::OpNode& node) -> of::Maybe<void> {
        // Build output tensors since batch size may changed
        of::LazyMode::Guard lazy_mode_disabled_guard{false};
        const oneflow::OperatorConf& op_conf = node.op().op_conf();
        if (op_conf.has_output_conf()) {
          oneflow::InterfaceBlobConf blob_conf = op_conf.output_conf().blob_conf();
          output_name_to_tensor_[op_conf.name()] =
              of::one::functional::Empty(
                  of::Shape(blob_conf.shape()),
                  of::DType::Get(static_cast<of::DataType>(blob_conf.data_type())).GetOrThrow(),
                  *device_.device_)
                  .GetPtrOrThrow();
        }
        return of::Maybe<void>::Ok();
      })
      .GetOrThrow();
}

void Graph::LoadCheckpoint() {}

void Graph::RegisterTensors() {
  std::vector<std::string> input_op_names;
  std::vector<std::shared_ptr<of::one::Tensor>> input_tensors;
  for (const auto& name_to_tensor : input_name_to_tensor_) {
    input_op_names.emplace_back(name_to_tensor.first);
    input_tensors.emplace_back(name_to_tensor.second);
  }

  std::vector<std::string> output_op_names;
  std::vector<std::shared_ptr<of::one::Tensor>> output_tensors;
  for (const auto& name_to_tensor : output_name_to_tensor_) {
    output_op_names.emplace_back(name_to_tensor.first);
    output_tensors.emplace_back(name_to_tensor.second);
  }

  std::vector<std::string> variable_op_names;
  std::vector<std::shared_ptr<of::one::Tensor>> variable_tensors;
  for (const auto& name_to_tensor : variable_op_name_to_tensor_) {
    variable_op_names.emplace_back(name_to_tensor.first);
    variable_tensors.emplace_back(name_to_tensor.second);
  }

  graph_->RegisterInputOpNamesAndTensors(input_op_names, input_tensors).GetOrThrow();
  graph_->RegisterVariableOpNamesAndTensors(variable_op_names, variable_tensors).GetOrThrow();
  graph_->RegisterOutputOpNamesAndTensors(output_op_names, output_tensors).GetOrThrow();

  output_tensor_tuple_ = ConvertToTensorTuple(output_tensors);
  parameter_tensor_tuple_ = ConvertToTensorTuple(variable_tensors);
}

Graph Load(const std::string& model_path, const Device& device) {
  Graph graph(model_path, device);
  return graph;
}

Graph Load(const std::string& model_path) {
  Device device = Device("cpu");
  return Load(model_path, device);
}

}  // namespace oneflow_api
