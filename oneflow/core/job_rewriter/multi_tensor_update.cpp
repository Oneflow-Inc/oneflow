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
#include "oneflow/core/common/util.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

struct SGDOptimizerKey {
  std::string learning_rate;
  double scale;
  float l1;
  float l2;
  float weight_decay;
  ParallelConf parallel_conf;
};

bool operator==(const SGDOptimizerKey& lhs, const SGDOptimizerKey& rhs) {
  return (lhs.learning_rate == rhs.learning_rate) && (lhs.scale == rhs.scale) && (lhs.l1 == rhs.l1)
         && (lhs.l2 == rhs.l2) && (lhs.weight_decay == rhs.weight_decay)
         && (lhs.parallel_conf == rhs.parallel_conf);
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::SGDOptimizerKey> {
  size_t operator()(const oneflow::SGDOptimizerKey& key) const {
    const auto& float_hash = std::hash<float>();
    const auto& double_hash = std::hash<float>();
    const auto& string_hash = std::hash<std::string>();
    const auto& parallel_conf_hash = std::hash<oneflow::ParallelConf>();

    return string_hash(key.learning_rate) ^ double_hash(key.scale) ^ float_hash(key.l1)
           ^ float_hash(key.l2) ^ float_hash(key.weight_decay)
           ^ parallel_conf_hash(key.parallel_conf);
  }
};

}  // namespace std

namespace oneflow {

namespace {

bool IsUserOpWithTypeName(const OperatorConf& op_conf, const std::string& op_type_name) {
  return op_conf.has_user_conf() && op_conf.user_conf().op_type_name() == op_type_name;
};

class MultiTensorUpdatePass final : public JobPass {
 public:
  MultiTensorUpdatePass() = default;
  ~MultiTensorUpdatePass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ParseBooleanFromEnv("ONEFLOW_ENABLE_MULTI_TENSOR_UPDATE", false);
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> MultiTensorUpdatePass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  if (!job_builder->job().job_conf().has_train_conf()) { return Maybe<void>::Ok(); }
  std::vector<OperatorConf> delete_ops;
  ParallelConf parallel_conf{};
  HashMap<SGDOptimizerKey, user_op::UserOpConfWrapperBuilder> multi_tensor_hashmap;

  op_graph.ForEachNode([&](OpNode* op_node) {
    const auto& op_conf = op_node->op().op_conf();
    if (!op_conf.has_variable_conf()) { return; }
    LogicalBlobId model_half_lbi;

    for (OpEdge* find_sgd_edge : op_node->out_edges()) {
      OpNode* find_sgd_update_node = find_sgd_edge->dst_node();
      if (!IsUserOpWithTypeName(find_sgd_update_node->op().op_conf(), "sgd_update")) { continue; }
      const user_op::UserOpConfWrapper sgd_user_conf(find_sgd_update_node->op().op_conf());
      // Currently only support for cuda, maybe remove this limit.
      if (find_sgd_update_node->parallel_desc().device_type() != DeviceType::kCUDA) { continue; }

      delete_ops.emplace_back(find_sgd_update_node->op().op_conf());
      parallel_conf = find_sgd_update_node->parallel_desc().parallel_conf();

      SGDOptimizerKey key{
          sgd_user_conf.input("learning_rate", 0),   sgd_user_conf.attr<double>("scale"),
          sgd_user_conf.attr<float>("l1"),           sgd_user_conf.attr<float>("l2"),
          sgd_user_conf.attr<float>("weight_decay"), parallel_conf};

      const auto& iter = multi_tensor_hashmap.find(key);

      if (iter != multi_tensor_hashmap.end()) {
        iter->second.Input("model", sgd_user_conf.input("model", 0))
            .Input("model_diff", sgd_user_conf.input("model_diff", 0));
      } else {
        user_op::UserOpConfWrapperBuilder multi_tensor_update_sgd_op_builder("multi_tensor_update");
        multi_tensor_update_sgd_op_builder.OpTypeName("multi_tensor_sgd_update")
            .Input("model", sgd_user_conf.input("model", 0))
            .Input("model_diff", sgd_user_conf.input("model_diff", 0))
            .Input("learning_rate", sgd_user_conf.input("learning_rate", 0))
            .Attr<double>("scale", sgd_user_conf.attr<double>("scale"))
            .Attr<float>("l1", sgd_user_conf.attr<float>("l1"))
            .Attr<float>("l2", sgd_user_conf.attr<float>("l2"))
            .Attr<float>("weight_decay", sgd_user_conf.attr<float>("weight_decay"));
        CHECK(sgd_user_conf.op_conf().has_scope_symbol_id());
        multi_tensor_update_sgd_op_builder.ScopeSymbolId(sgd_user_conf.op_conf().scope_symbol_id());
        multi_tensor_hashmap.emplace(key, multi_tensor_update_sgd_op_builder);
      }
      break;
    }
  });
  for (auto& op : multi_tensor_hashmap) {
    auto multi_tensor_update_sgd_op = op.second.Build();
    job_builder->AddOps(parallel_conf, {multi_tensor_update_sgd_op.op_conf()});
  }
  job_builder->DelOps(delete_ops);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("MultiTensorUpdatePass", MultiTensorUpdatePass);

}  // namespace oneflow
