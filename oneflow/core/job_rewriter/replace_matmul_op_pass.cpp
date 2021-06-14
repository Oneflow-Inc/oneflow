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

void GetMatmulParallelDistributionSignature(bool transpose_a, bool transpose_b,
                                            ParallelDistributionSignature* signature) {
  ParallelDistribution a_distribution;
  ParallelDistribution b_distribution;
  ParallelDistribution out_distribution;
  if (!transpose_a && !transpose_b) {
    a_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    a_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
    b_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
    b_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
    out_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    out_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
  } else if (!transpose_a && transpose_b) {
    a_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    a_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
    b_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
    b_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
    out_distribution.add_sbp_parallel()->mutable_partial_sum_parallel();
    out_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
  } else if (transpose_a && !transpose_b) {
    a_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    a_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
    b_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    b_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
    out_distribution.add_sbp_parallel()->mutable_partial_sum_parallel();
    out_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
  } else {
    UNIMPLEMENTED();
  }
  (*signature->mutable_bn_in_op2parallel_distribution())["a_0"] = a_distribution;
  (*signature->mutable_bn_in_op2parallel_distribution())["b_0"] = b_distribution;
  (*signature->mutable_bn_in_op2parallel_distribution())["out_0"] = out_distribution;
}

}  // namespace

class ReplaceMatmulOpPass final : public JobPass {
 public:
  ReplaceMatmulOpPass() = default;
  ~ReplaceMatmulOpPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> ReplaceMatmulOpPass::Apply(Job* job, JobPassCtx* ctx) const {
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  HashMap<std::string, OperatorConf> name2op_conf;
  HashMap<std::string, ParallelDistributionSignature> op_name2parallel_distribution_signature;
  auto GetOperatorConf4Modify = [&name2op_conf](const OperatorConf& op_conf) {
    const auto& it = name2op_conf.find(op_conf.name());
    if (it != name2op_conf.end()) {
      return &it->second;
    } else {
      name2op_conf[op_conf.name()] = op_conf;
      return &name2op_conf.at(op_conf.name());
    }
  };
  std::vector<std::string> cast_parallel_distribution;
  cast_parallel_distribution.push_back("S(0)");
  cast_parallel_distribution.push_back("S(1)");

  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name != "summa_matmul_placeholder") { return; }
    if (op_node->parallel_desc().hierarchy()->NumAxes() != 2) { return; }
    const user_op::UserOpConfWrapper user_op_conf(op_conf);
    bool transpose_a = user_op_conf.attr<bool>("transpose_a");
    bool transpose_b = user_op_conf.attr<bool>("transpose_b");
    if (transpose_a && transpose_b) { UNIMPLEMENTED(); }
    ParallelDistributionSignature signature;
    GetMatmulParallelDistributionSignature(transpose_a, transpose_b, &signature);
    op_name2parallel_distribution_signature[op_node->op().op_name()] = signature;
    OperatorConf* new_op_conf = GetOperatorConf4Modify(op_node->op().op_conf());
    new_op_conf->mutable_user_conf()->set_op_type_name("matmul");
    const int64_t scope_symbol_id = op_node->op().op_conf().scope_symbol_id();
    auto parallel_cast_op =
        user_op::UserOpConfWrapperBuilder("Matmul_cast_" + NewUniqueId())
            .Op("hierarchical_parallel_cast")
            .Input("in", user_op_conf.output("out", 0))
            .Output("out")
            .Attr<std::vector<std::string>>("parallel_distribution", cast_parallel_distribution)
            .Attr<std::string>("grad_mode", "auto")
            .Attr<std::vector<std::string>>("grad_parallel_distribution",
                                            std::vector<std::string>())
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder.AddOps(op_node->parallel_desc().parallel_conf(), {parallel_cast_op.op_conf()});

    for (const OpEdge* op_edge : op_node->out_edges()) {
      for (const LogicalBlobId& lbi : op_edge->lbis()) {
        const OpNode* dst_node = op_edge->dst_node();
        for (const std::string& ibn : op_edge->lbi2ibns().at(lbi)) {
          OperatorConf* new_dst_op_conf = GetOperatorConf4Modify(dst_node->op().op_conf());
          std::string old_lbn = ReplaceInputLbnInOpCustomizedConf(
              new_dst_op_conf, ibn, parallel_cast_op.output("out", 0));
          CHECK(old_lbn == GenLogicalBlobName(lbi));
        }
      }
    }
  });
  for (const auto& pair : name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }
  for (const auto& pair : op_name2parallel_distribution_signature) {
    job_builder.AddParallelDistributionSignature4OpName(pair.first, pair.second);
  }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("ReplaceMatmulOpPass", ReplaceMatmulOpPass);

}  // namespace oneflow
