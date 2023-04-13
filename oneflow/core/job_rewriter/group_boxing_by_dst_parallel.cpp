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
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

const Scope& Scope4ScopeSymbolId(int64_t scope_symbol_id) {
  CHECK(Singleton<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id));
  return Singleton<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
}

const Scope& Scope4OpNode(const OpNode* op_node) {
  const OperatorConf& op_conf = op_node->op().op_conf();
  CHECK(op_conf.has_scope_symbol_id());
  return Scope4ScopeSymbolId(op_conf.scope_symbol_id());
}

bool OpNodeHasScope(const OpNode* node) { return node->op().op_conf().has_scope_symbol_id(); }

int64_t GetStageIdHint(const OpNode* node) {
  return Scope4OpNode(node).Int64("pipeline_stage_id_hint");
}

Maybe<void> GroupBoxingByDstParallel(const OpGraph& op_graph, JobBuilder* job_builder) {
  {
    // NOTE(chengcheng): Disable group boxing for pipeline parallel, because there will be bad case
    //  make forward backward exec sequential in ZeRO + 3-D Parallel by insert additional boxing
    //  identity.
    int64_t max_stage_id = 0;
    op_graph.ForEachNode([&](const OpNode* this_node) {
      if (!OpNodeHasScope(this_node)) {
        LOG(WARNING) << " op : " << this_node->op().op_conf().DebugString() << " has NOT scope!";
        return;
      }
      max_stage_id = std::max(max_stage_id, GetStageIdHint(this_node));
    });
    if (max_stage_id > 0) { return Maybe<void>::Ok(); }
  }
  HashMap<LogicalBlobId, HashMap<std::pair<ParallelDesc, NdSbp>,
                                 std::vector<std::pair<const OpNode*, std::string>>>>
      lbi2consumer_grouped_by_parallel;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  op_graph.ForEachNode([&](const OpNode* node) {
    OperatorConf::OpTypeCase op_type_case = node->op().op_conf().op_type_case();
    if (IsClassRegistered<int32_t, DisableInputBoxingGroup>(op_type_case)) { return; }
    for (const std::string& ibn : node->op().input_bns()) {
      const auto& blob_modifier_ = node->op().InputBlobModifier4Ibn(ibn);
      if (blob_modifier_.has_is_mutable() && blob_modifier_.is_mutable()) { continue; }
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      const OpNode& producer = node->ProducerOpNode4Lbi(lbi);
      const auto& logical_shape = node->LogicalBlobDesc4Lbi(lbi).shape();
      const NdSbp& producer_nd_sbp = producer.NdSbp4Lbi(lbi);
      const std::string& producer_lbn = *CHECK_JUST(producer.op().obn4lbi(lbi));
      const ParallelDesc& producer_parallel_desc =
          *CHECK_JUST(producer.op().GetParallelDesc4BnInOp(producer_lbn)).get();
      ParallelDesc reduced_in_parallel_desc = producer_parallel_desc;
      NdSbp reduced_in_nd_sbp;
      NdSbpDimReduce(producer_parallel_desc, producer_nd_sbp, &reduced_in_parallel_desc,
                     &reduced_in_nd_sbp, logical_shape);

      const NdSbp& consumer_nd_sbp = node->NdSbp4BnInOp(ibn);
      const ParallelDesc& consumer_parallel_desc =
          *CHECK_JUST(node->op().GetParallelDesc4BnInOp(ibn));
      ParallelDesc reduced_out_parallel_desc = consumer_parallel_desc;
      NdSbp reduced_out_nd_sbp;
      NdSbpDimReduce(consumer_parallel_desc, consumer_nd_sbp, &reduced_out_parallel_desc,
                     &reduced_out_nd_sbp, logical_shape);

      if (reduced_in_parallel_desc == reduced_out_parallel_desc
          && reduced_in_nd_sbp == reduced_out_nd_sbp) {
        continue;
      }
      lbi2consumer_grouped_by_parallel[lbi][{reduced_out_parallel_desc, reduced_out_nd_sbp}]
          .push_back({node, ibn});
      if (op_node2op_conf.find(node) == op_node2op_conf.end()) {
        op_node2op_conf[node] = node->op().op_conf();
      }
    }
  });
  for (const auto& lbi7groups : lbi2consumer_grouped_by_parallel) {
    const LogicalBlobId& lbi = lbi7groups.first;
    for (const auto& parallel7group : lbi7groups.second) {
      if (parallel7group.second.size() < 2) { continue; }
      const ParallelDesc& dst_parallel_desc = parallel7group.first.first;
      const NdSbp& dst_nd_sbp = parallel7group.first.second;
      OperatorConf identity_op_conf{};
      identity_op_conf.set_name("Sys-Boxing-GroupIdentity-" + lbi.op_name() + "_" + lbi.blob_name()
                                + "-" + NewUniqueId());
      IdentityOpConf* identity_conf = identity_op_conf.mutable_identity_conf();
      identity_conf->set_in(GenLogicalBlobName(lbi));
      identity_conf->set_out("out");
      job_builder->AddOps(dst_parallel_desc.parallel_conf(), {identity_op_conf});
      NdSbpSignature identity_nd_sbp_signature;
      (*identity_nd_sbp_signature.mutable_bn_in_op2nd_sbp())["in"] = dst_nd_sbp;
      (*identity_nd_sbp_signature.mutable_bn_in_op2nd_sbp())["out"] = dst_nd_sbp;
      job_builder->AddNdSbpSignature4OpName(identity_op_conf.name(), identity_nd_sbp_signature);

      LogicalBlobId grouped_lbi;
      grouped_lbi.set_op_name(identity_op_conf.name());
      grouped_lbi.set_blob_name(identity_conf->out());
      for (const auto& consumer7ibn : parallel7group.second) {
        const OpNode* consumer = consumer7ibn.first;
        const std::string& ibn = consumer7ibn.second;
        OperatorConf& consumer_op_conf = op_node2op_conf[consumer];
        const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn,
                                                                GenLogicalBlobName(grouped_lbi));
        CHECK_EQ_OR_RETURN(GenLogicalBlobName(lbi), old_val);
      }
    }
  }
  for (const auto& op_node7op_conf : op_node2op_conf) {
    JUST(job_builder->MutOpOnlyOnce(op_node7op_conf.second));
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
