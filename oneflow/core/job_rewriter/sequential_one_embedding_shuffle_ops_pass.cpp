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

namespace oneflow {

namespace {

class SequentialOneEmbeddingOpsPass final : public JobPass {
 public:
  SequentialOneEmbeddingOpsPass() = default;
  ~SequentialOneEmbeddingOpsPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false);
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> SequentialOneEmbeddingOpsPass::Apply(const OpGraph& op_graph,
                                                 JobBuilder* job_builder) const {
  HashMap<std::string, std::vector<std::string>> stream_name_hint2shuffle_op_names;
  op_graph.TopoForEachNode([&](const OpNode* op_node) {
    if (!(IsUserOpWithTypeName(op_node->op().op_conf(), "id_shuffle")
          || IsUserOpWithTypeName(op_node->op().op_conf(), "embedding_shuffle")
          || IsUserOpWithTypeName(op_node->op().op_conf(), "embedding_gradient_shuffle"))) {
      return;
    }
    OperatorConf op_conf = op_node->op().op_conf();
    std::string stream_name;
    if (op_conf.has_stream_name_hint()) {
      stream_name = op_conf.stream_name_hint();
    } else {
      stream_name = "DEFAULT";
    }
    const auto& it = stream_name_hint2shuffle_op_names.find(stream_name);
    if (it != stream_name_hint2shuffle_op_names.end()) {
      if (it->second.size() > 0) {
        std::string pre_shuffle_op_name = it->second.back();
        op_conf.add_ctrl_in_op_name(pre_shuffle_op_name);
        job_builder->MutOpsOnlyOnce({op_conf});
      }
      it->second.push_back(op_conf.name());
    } else {
      std::vector<std::string> shuffle_ops{op_conf.name()};
      CHECK(stream_name_hint2shuffle_op_names.emplace(stream_name, shuffle_ops).second);
    }
  });

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("SequentialOneEmbeddingOpsPass", SequentialOneEmbeddingOpsPass);

}  // namespace oneflow
