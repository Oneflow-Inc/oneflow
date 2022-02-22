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
#include "oneflow/core/job_rewriter/boxing_with_middle_nodes.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/auto_parallel/boxing_collector.h"

namespace oneflow {

Maybe<void> BoxingWithMiddleNodes(const OpGraph& op_graph, JobBuilder* job_builder) {
  // Not allowed two-step boxing and disable checking for debugging
  if (ParseBooleanFromEnv("ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK", false)) {
    return Maybe<void>::Ok();
  }
  // Initialize boxing collector
  BoxingCollector boxing_collector;
  // We assemble the boxing table from S(0) to S(5).
  // Those splitting in higher axes are considered in the customized boxing.
  constexpr int32_t kRegularMaxSplitAxes = 6;
  JUST(boxing_collector.Init(kRegularMaxSplitAxes));
  std::vector<NdSbp> middle_sbps;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  // Fill other unsupported combinations
  op_graph.ForEachNode([&](const OpNode* node) -> Maybe<void> {
    OperatorConf::OpTypeCase op_type_case = node->op().op_conf().op_type_case();
    if (IsClassRegistered<int32_t, DisableInputBoxingGroup>(op_type_case)) {
      return Maybe<void>::Ok();
    }
    for (const std::string& ibn : node->op().input_bns()) {
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      const OpNode& producer = node->ProducerOpNode4Lbi(lbi);
      const NdSbp& producer_nd_sbp = producer.NdSbp4Lbi(lbi);
      const NdSbp& consumer_nd_sbp = node->NdSbp4BnInOp(ibn);

      // Needs more effort if dealing with different placement
      if (node->parallel_desc().parallel_num() != 1 && producer_nd_sbp != consumer_nd_sbp) {
        const auto& logical_blob_desc = producer.LogicalBlobDesc4Lbi(lbi);
        // Ask for middle nodes
        JUST(boxing_collector.AskSbpCombination(
            producer_nd_sbp, consumer_nd_sbp, logical_blob_desc, producer.parallel_desc(),
            node->parallel_desc(), /*is_customized=*/false, middle_sbps, /*compute_cost=*/false));
        // move to the next ibn if no middle nodes needed
        if (middle_sbps.size() <= 0) { continue; }
        LogicalBlobId middle_node_lbi = lbi;
        for (int32_t middle_node_id = 0; middle_node_id < middle_sbps.size(); middle_node_id++) {
          // Create the middle operators
          OperatorConf identity_op_conf{};
          identity_op_conf.set_name("System-Boxing-Middle-Identity-" + NewUniqueId());
          IdentityOpConf* identity_conf = identity_op_conf.mutable_identity_conf();
          identity_conf->set_in(GenLogicalBlobName(middle_node_lbi));
          identity_conf->set_out("out");
          job_builder->AddOps(node->parallel_desc().parallel_conf(), {identity_op_conf});
          NdSbpSignature identity_nd_sbp_signature;
          (*identity_nd_sbp_signature.mutable_bn_in_op2nd_sbp())["in"] =
              middle_sbps[middle_node_id];
          (*identity_nd_sbp_signature.mutable_bn_in_op2nd_sbp())["out"] =
              middle_sbps[middle_node_id];
          job_builder->AddNdSbpSignature4OpName(identity_op_conf.name(), identity_nd_sbp_signature);
          // Connection for the next middle node
          middle_node_lbi.set_op_name(identity_op_conf.name());
          middle_node_lbi.set_blob_name(identity_conf->out());
        }
        // Replace input blob with configuration from middle nodes
        if (op_node2op_conf.find(node) == op_node2op_conf.end()) {
          op_node2op_conf[node] = node->op().op_conf();
        }
        OperatorConf& consumer_op_conf = op_node2op_conf[node];
        const auto& old_val = ReplaceInputLbnInOpCustomizedConf(
            &consumer_op_conf, ibn, GenLogicalBlobName(middle_node_lbi));
        CHECK_EQ_OR_RETURN(GenLogicalBlobName(lbi), old_val);
      }
    }

    return Maybe<void>::Ok();
  });
  for (const auto& op_node7op_conf : op_node2op_conf) {
    JUST(job_builder->MutOpOnlyOnce(op_node7op_conf.second));
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
