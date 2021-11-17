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
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cuda_stream_index.h"

namespace oneflow {

class EmbeddingOpsAddStreamIndexHintPass final : public JobPass {
 public:
  EmbeddingOpsAddStreamIndexHintPass() = default;
  ~EmbeddingOpsAddStreamIndexHintPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return true;
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> EmbeddingOpsAddStreamIndexHintPass::Apply(const OpGraph& op_graph,
                                                      JobBuilder* job_builder) const {
  CudaStreamIndexGenerator stream_idx_gen;
  int32_t stream_index = static_cast<int32_t>(stream_idx_gen.GenerateNamedStreamIndex("EMBEDDING"));

  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name != "embedding_prefetch" && op_type_name != "embedding_lookup" && op_type_name != "sgd_embedding_update") {
      return;
    }
    OperatorConf new_op_conf = op_conf;
    new_op_conf.set_stream_index_hint(stream_index);
    job_builder->MutOpsOnlyOnce({new_op_conf});
  });
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("EmbeddingOpsAddStreamIndexHintPass", EmbeddingOpsAddStreamIndexHintPass);

}  // namespace oneflow
