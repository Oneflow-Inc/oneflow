#include "oneflow/core/job_completer/aggregate_boxing_by_dst_parallel.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void AggregateBoxingByDstParallel(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  HashMap<LogicalBlobId, HashMap<std::pair<ParallelDesc, SbpParallel>,
                                 std::vector<std::pair<const OpNode*, std::string>>>>
      lbi2grouped_consumers;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  op_graph.ForEachNode([&](const OpNode* node) {
    for (const std::string& ibn : node->op().input_bns()) {
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      const OpNode* producer = node->ProducerOpNode4Lbi(lbi);
      const SbpParallel& producer_sbp = producer->SbpParallel4Lbi(lbi);
      const SbpParallel& consumer_sbp = node->SbpParallel4BnInOp(ibn);
      if (producer->parallel_desc() != node->parallel_desc() || producer_sbp != consumer_sbp) {
        lbi2grouped_consumers[lbi][{node->parallel_desc(), consumer_sbp}].push_back({node, ibn});
        if (op_node2op_conf.find(node) == op_node2op_conf.end()) {
          op_node2op_conf[node] = node->op().op_conf();
        }
      }
    }
  });
  for (const auto& lbi7groups : lbi2grouped_consumers) {
    const LogicalBlobId& lbi = lbi7groups.first;
    for (const auto& parallel7group : lbi7groups.second) {
      if (parallel7group.second.size() < 2) { continue; }
      const ParallelDesc& dst_parallel_desc = parallel7group.first.first;
      const SbpParallel& dst_sbp_parallel = parallel7group.first.second;
      OperatorConf identity_op_conf{};
      identity_op_conf.set_name("System-Boxing-Identity-" + NewUniqueId());
      IdentityOpConf* identity_conf = identity_op_conf.mutable_identity_conf();
      identity_conf->set_in(GenLogicalBlobName(lbi));
      identity_conf->set_out("out");
      job_builder.AddOps(dst_parallel_desc.parallel_conf(), {identity_op_conf});
      SbpSignature identity_sbp_signature;
      (*identity_sbp_signature.mutable_bn_in_op2sbp_parallel())["in"] = dst_sbp_parallel;
      (*identity_sbp_signature.mutable_bn_in_op2sbp_parallel())["out"] = dst_sbp_parallel;
      (*job->mutable_sbp_conf()->mutable_op_name2sbp_signature_conf())[identity_op_conf.name()] =
          identity_sbp_signature;
      LogicalBlobId aggregated_lbi;
      aggregated_lbi.set_op_name(identity_op_conf.name());
      aggregated_lbi.set_blob_name(identity_conf->out());
      for (const auto& consumer7ibn : parallel7group.second) {
        const OpNode* consumer = consumer7ibn.first;
        const std::string& ibn = consumer7ibn.second;
        OperatorConf& consumer_op_conf = op_node2op_conf[consumer];
        PbMessage* consumer_op_type_conf =
            MutableMessageInPbMessage(&consumer_op_conf, consumer_op_conf.op_type_case());
        SetBnValInOpTypeConf(consumer_op_type_conf, ibn, GenLogicalBlobName(lbi),
                             GenLogicalBlobName(aggregated_lbi));
      }
    }
  }
  for (const auto& op_node7op_conf : op_node2op_conf) {
    job_builder.MutOps({op_node7op_conf.second});
  }
}

}  // namespace oneflow
