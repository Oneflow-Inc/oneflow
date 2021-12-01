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
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/api/common/job_build_and_infer_ctx.h"
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_ir.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/operator/interface_blob_conf.pb.h"

namespace oneflow_api {

namespace of = oneflow;

class CompileScope {
 public:
  CompileScope(const of::JobConfigProto& job_config) {
    std::shared_ptr<of::Scope> scope = of::MakeInitialScope();
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
  of::LazyMode::Guard lazy_mode_enabled_guard{/*is_enabled*/ true};
};

Graph::Graph(const std::string& model_path, const Device& device) {
  // TODO(zzk0): model_path is a directory, need to concatenate filename
  of::LoadJobFromIR(&job_, model_path).GetOrThrow();
  graph_ = std::make_shared<of::NNGraph>(job_.job_conf().job_name());
}

Graph::Graph(const std::string& model_path) : Graph(model_path, Device("cpu")) {}

std::vector<Tensor> Graph::forward(const std::vector<Tensor>& inputs) {
  if (!is_compiled_) {
    Compile();
    is_compiled_ = true;
  }
  return Run(inputs);
}

void Graph::Compile() {
  BuildGraph();
  RegisterTensors();
  graph_->CompileAndInitRuntime().GetOrThrow();
}

void Graph::BuildGraph() {
  CompileScope build_graph_scope(job_.job_conf());

  of::OpGraph op_graph(job_);
  op_graph
      .ForEachOpNode([&](const of::OpNode& node) -> of::Maybe<void> {
        // add op
        auto* ctx = of::GetCurInferCtx().GetOrThrow();
        ctx->AddAndInferConsistentOp(node.op().op_conf()).GetOrThrow();

        // input tensor
        if (node.op().op_conf().has_input_conf()) {}

        // create variable op tensor
        if (node.op().op_conf().has_variable_conf()) {
          // TODO(zzk0): load from local path
          // variable_op_name_to_tensor_[node.op().op_name()] =
          // of::one::MirroredTensor::MakeTensor();
        }

        // output tensor
        if (node.op().op_conf().has_output_conf()) {
          oneflow::InterfaceBlobConf blob_conf = node.op().op_conf().output_conf().blob_conf();
          blob_conf.shape();
          // oneflow::one::functional::Empty()
        }

        return of::Maybe<void>::Ok();
      })
      .GetOrThrow();

  // CurJobBuildAndInferCtx_Complete
  of::CurJobBuildAndInferCtx_Complete().GetOrThrow();
}

void Graph::RegisterTensors() {
  // graph_->RegisterInputOpNamesAndTensors
  // graph_->RegisterOutputOpNamesAndTensors
  // graph_->RegisterVariableOpNamesAndTensors
}

std::vector<Tensor> Run(const std::vector<Tensor>& inputs) {
  // RunLazyNNGraph && SoftSyncNNGraphBuffers
  return std::vector<Tensor>{};
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
