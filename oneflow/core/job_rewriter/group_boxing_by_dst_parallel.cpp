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
#include "oneflow/core/job_rewriter/group_boxing_by_dst_parallel.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void GroupBoxingByDstParallel(const OpGraph& op_graph, JobBuilder* job_builder) {
  HashMap<LogicalBlobId, HashMap<std::pair<ParallelDesc, ParallelDistribution>,
                                 std::vector<std::pair<const OpNode*, std::string>>>>
      lbi2consumer_grouped_by_parallel_distribution;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  op_graph.ForEachNode([&](const OpNode* node) {
    OperatorConf::OpTypeCase op_type_case = node->op().op_conf().op_type_case();
    if (IsClassRegistered<int32_t, DisableInputBoxingGroup>(op_type_case)) { return; }
    for (const std::string& ibn : node->op().input_bns()) {
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      const OpNode& producer = node->ProducerOpNode4Lbi(lbi);
      const ParallelDistribution& producer_parallel_distribution =
          producer.ParallelDistribution4Lbi(lbi);
      const ParallelDistribution& consumer_parallel_distribution =
          node->ParallelDistribution4BnInOp(ibn);

      if (producer.parallel_desc() != node->parallel_desc()
          || (node->parallel_desc().parallel_num() != 1
              && producer_parallel_distribution != consumer_parallel_distribution)) {
        lbi2consumer_grouped_by_parallel_distribution[lbi][{node->parallel_desc(),
                                                            consumer_parallel_distribution}]
            .push_back({node, ibn});
        if (op_node2op_conf.find(node) == op_node2op_conf.end()) {
          op_node2op_conf[node] = node->op().op_conf();
        }
      }
    }
  });
  for (const auto& lbi7groups : lbi2consumer_grouped_by_parallel_distribution) {
    const LogicalBlobId& lbi = lbi7groups.first;
    for (const auto& parallel7group : lbi7groups.second) {
      if (parallel7group.second.size() < 2) { continue; }
      const ParallelDesc& dst_parallel_desc = parallel7group.first.first;
      const ParallelDistribution& dst_parallel_distribution = parallel7group.first.second;
      OperatorConf identity_op_conf{};
      identity_op_conf.set_name("System-Boxing-Identity-" + NewUniqueId());
      IdentityOpConf* identity_conf = identity_op_conf.mutable_identity_conf();
      identity_conf->set_in(GenLogicalBlobName(lbi));
      identity_conf->set_out("out");
      job_builder->AddOps(dst_parallel_desc.parallel_conf(), {identity_op_conf});
      ParallelDistributionSignature identity_parallel_distribution_signature;
      (*identity_parallel_distribution_signature.mutable_bn_in_op2parallel_distribution())["in"] =
          dst_parallel_distribution;
      (*identity_parallel_distribution_signature.mutable_bn_in_op2parallel_distribution())["out"] =
          dst_parallel_distribution;
      job_builder->AddParallelDistributionSignature4OpName(
          identity_op_conf.name(), identity_parallel_distribution_signature);

      LogicalBlobId grouped_lbi;
      grouped_lbi.set_op_name(identity_op_conf.name());
      grouped_lbi.set_blob_name(identity_conf->out());
      for (const auto& consumer7ibn : parallel7group.second) {
        const OpNode* consumer = consumer7ibn.first;
        const std::string& ibn = consumer7ibn.second;
        OperatorConf& consumer_op_conf = op_node2op_conf[consumer];
        const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn,
                                                                GenLogicalBlobName(grouped_lbi));
        CHECK_EQ(GenLogicalBlobName(lbi), old_val);
      }
    }
  }
  for (const auto& op_node7op_conf : op_node2op_conf) {
    job_builder->MutOpsOnlyOnce({op_node7op_conf.second});
  }
}

}  // namespace oneflow
