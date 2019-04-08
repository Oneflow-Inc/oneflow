#include "oneflow/core/job_completer/all_reduce_add_pass.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

namespace {

std::function<const OpNode*(const LogicalBlobId&)> MakeGetterProducerOpNode4Lbi(
    const OpGraph& op_graph) {
  auto lbi2producer_op_node = std::make_shared<HashMap<LogicalBlobId, const OpNode*>>();
  op_graph.ForEachNode([&](OpNode* op_node) {
    for (const std::string& obn : op_node->op().output_bns()) {
      CHECK(lbi2producer_op_node->emplace(op_node->op().BnInOp2Lbi(obn), op_node).second);
    }
  });
  return [lbi2producer_op_node](const LogicalBlobId& lbi) -> const OpNode* {
    return lbi2producer_op_node->at(lbi);
  };
}

std::function<const OpNode*(const LogicalBlobId&)> MakeGetterSoleConsumerOpNode4Lbi(
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi) {
  return [ProducerOpNode4Lbi](const LogicalBlobId& lbi) -> const OpNode* {
    const OpNode* producer = ProducerOpNode4Lbi(lbi);
    const OpEdge* consumer_edge = nullptr;
    for (const OpEdge* edge : producer->out_edges()) {
      if (std::find(edge->lbis().begin(), edge->lbis().end(), lbi) != edge->lbis().end()) {
        CHECK(consumer_edge == nullptr);
        consumer_edge = edge;
      }
    }
    CHECK_NOTNULL(consumer_edge);
    return consumer_edge->dst_node();
  };
}

LogicalBlobId FindPartialSum2BroadcastSoleConsumerLbi(
    const LogicalBlobId& lbi,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::function<const OpNode*(const LogicalBlobId&)>& SoleConsumerOpNode4Lbi) {
  TODO();
  return lbi;
}

void FindAllReducedLbis(
    const Job& job, const OpGraph& op_graph,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    std::vector<LogicalBlobId>* lbis) {
  auto SoleConsumerOpNode4Lbi = MakeGetterSoleConsumerOpNode4Lbi(ProducerOpNode4Lbi);
  TODO();
}

void SortAllReducedLbis(
    const OpGraph& op_graph,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    std::vector<LogicalBlobId>* lbis) {
  TODO();
}

void GroupAllReducedLbisByStrategy(const OpGraph& op_graph,
                                   const std::vector<LogicalBlobId>& sorted_lbis,
                                   std::vector<std::vector<LogicalBlobId>>* lbi_groups) {
  TODO();
}

void GroupAllReducedLbis(
    const Job& job, const OpGraph& op_graph,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    std::vector<std::vector<LogicalBlobId>>* lbi_groups) {
  std::vector<LogicalBlobId> lbis;
  FindAllReducedLbis(job, op_graph, ProducerOpNode4Lbi, &lbis);
  SortAllReducedLbis(op_graph, ProducerOpNode4Lbi, &lbis);
  GroupAllReducedLbisByStrategy(op_graph, lbis, lbi_groups);
}

void AddReduceConcatAndReduceIdentityOpConf(
    const JobBuilder& job_builder,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& lbi_groups, LogicalBlobId* grouped_lbi) {
  TODO();
}

void AddAllReduceOpConf(const JobBuilder& job_builder, const LogicalBlobId& grouped_lbi,
                        LogicalBlobId* all_reduced_lbi) {
  // TODO: support all type all reduce
  OperatorConf all_reduce_op;
  all_reduce_op.set_name("System-Boxing-AllReduce-" + grouped_lbi.op_name() + "-"
                         + grouped_lbi.blob_name());
  NcclAllReduceOpConf* nccl_all_reduce_op_conf = all_reduce_op.mutable_nccl_all_reduce_conf();
  nccl_all_reduce_op_conf->set_in(GenLogicalBlobName(grouped_lbi));
  nccl_all_reduce_op_conf->set_out("out");
  all_reduced_lbi->set_op_name(all_reduce_op.name());
  all_reduced_lbi->set_blob_name(nccl_all_reduce_op_conf->out());
  // TODO: all all reduce op to job_builder, need `parallel_desc`
}

void AddReduceSplitOpConf(
    const JobBuilder& job_builder,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& lbi_groups, const LogicalBlobId& all_reduced_lbi) {
  TODO();
}

void BuildAllReduceStruct(
    const JobBuilder& job_builder,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& lbi_groups) {
  LogicalBlobId grouped_lbi;
  AddReduceConcatAndReduceIdentityOpConf(job_builder, ProducerOpNode4Lbi, lbi_groups, &grouped_lbi);
  LogicalBlobId all_reduced_lbi;
  AddAllReduceOpConf(job_builder, grouped_lbi, &all_reduced_lbi);
  AddReduceSplitOpConf(job_builder, ProducerOpNode4Lbi, lbi_groups, all_reduced_lbi);
}

}  // namespace

void AllReduceAddPass::Apply(Job* job) const {
  OpGraph op_graph(*job);
  auto ProducerOpNode4Lbi = MakeGetterProducerOpNode4Lbi(op_graph);
  std::vector<std::vector<LogicalBlobId>> lbi_groups;
  GroupAllReducedLbis(*job, op_graph, ProducerOpNode4Lbi, &lbi_groups);
  JobBuilder job_builder(job);
  for (const auto& lbi_group : lbi_groups) {
    BuildAllReduceStruct(job_builder, ProducerOpNode4Lbi, lbi_group);
  }
}

}  // namespace oneflow
