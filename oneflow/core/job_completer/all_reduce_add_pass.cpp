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

LogicalBlobId FindP2BLbiWithSoleConsumer(
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
  const auto& lbi2diff_lbi = job.helper().tag2lbi_relations().at(kProducedLbi2ConsumedDiffLbi);
  HashSet<LogicalBlobId> key_check;
  HashSet<LogicalBlobId> value_check;
  HashSet<LogicalBlobId> diff_lbi_check;
  for (const auto& pair : lbi2diff_lbi.pair()) {
    CHECK(key_check.emplace(pair.first()).second);
    CHECK(value_check.emplace(pair.second()).second);
    const auto* producer = ProducerOpNode4Lbi(pair.first());
    if (producer->parallel_desc().parallel_num() == 1) { continue; }
    if (producer->op().op_conf().has_variable_conf() == false) { continue; }
    if (producer->SbpParallel4Lbi(pair.first()).has_broadcast_parallel() == false) { continue; }
    const auto& diff_lbi =
        FindP2BLbiWithSoleConsumer(pair.second(), ProducerOpNode4Lbi, SoleConsumerOpNode4Lbi);
    lbis->push_back(diff_lbi);
    CHECK(diff_lbi_check.emplace(diff_lbi).second);
  }
}

std::function<int32_t(const OpNode*)> MakeGetterDepth4OpNode(const OpGraph& op_graph) {
  auto op_node2depth = std::make_shared<HashMap<const OpNode*, int32_t>>();
  int32_t depth = 0;
  op_graph.TopoForEachNode([&](OpNode* op_node) { op_node2depth->emplace(op_node, depth++); });
  return [op_node2depth](const OpNode* op_node) { return op_node2depth->at(op_node); };
}

void SortAllReducedLbis(
    const OpGraph& op_graph,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    std::vector<LogicalBlobId>* lbis) {
  auto Depth4OpNode = MakeGetterDepth4OpNode(op_graph);
  std::sort(lbis->begin(), lbis->end(), [&](const LogicalBlobId& lhs, const LogicalBlobId& rhs) {
    return Depth4OpNode(ProducerOpNode4Lbi(lhs)) > Depth4OpNode(ProducerOpNode4Lbi(rhs));
  });
}

void ForEachLbisGroupByDataTypeAndParallelDesc(
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& sorted_lbis,
    const std::function<void(const std::vector<LogicalBlobId>&)>& Handler) {
  HashMap<int32_t, HashMap<ParallelDesc, std::vector<LogicalBlobId>>> dtype2parellel_desc2op_nodes;
  for (const LogicalBlobId& lbi : sorted_lbis) {
    const OpNode* producer = ProducerOpNode4Lbi(lbi);
    DataType dtype = producer->LogicalBlobDesc4Lbi(lbi).data_type();
    dtype2parellel_desc2op_nodes[dtype][producer->parallel_desc()].push_back(lbi);
  }
  for (const auto& out_pair : dtype2parellel_desc2op_nodes) {
    for (const auto& pair : out_pair.second) { Handler(pair.second); }
  }
}

void GroupAllReducedLbisByStrategy(
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& sorted_lbis,
    std::vector<std::vector<LogicalBlobId>>* lbi_groups) {
  auto MemSize4Lbi = [&](const LogicalBlobId& lbi) -> size_t {
    const OpNode* producer = ProducerOpNode4Lbi(lbi);
    const BlobDesc& logical_blob_desc = producer->LogicalBlobDesc4Lbi(lbi);
    int64_t elem_cnt = logical_blob_desc.shape().elem_cnt();
    size_t model_size = elem_cnt * GetSizeOfDataType(logical_blob_desc.data_type());
    return RoundUp(model_size, kCudaAlignSize);
  };
  ForEachLbisGroupByDataTypeAndParallelDesc(
      ProducerOpNode4Lbi, sorted_lbis, [&](const std::vector<LogicalBlobId>& lbis) {
        size_t model_total_size = 0;
        for (const auto& lbi : lbis) { model_total_size += MemSize4Lbi(lbi); }
        size_t avg_size = model_total_size / Global<JobDesc>::Get()->all_reduce_group_num();
        const size_t group_min_size = Global<JobDesc>::Get()->all_reduce_group_min_byte();
        const float group_size_warmup = Global<JobDesc>::Get()->all_reduce_group_size_warmup();
        size_t cur_group_size = group_min_size / group_size_warmup;
        auto GetCurGroupSize = [&]() {
          if (cur_group_size < avg_size) { cur_group_size *= group_size_warmup; }
          return std::min(cur_group_size, avg_size);
        };
        lbi_groups->emplace_back(std::vector<LogicalBlobId>{});
        size_t cur_group_model_size = 0;
        for (const LogicalBlobId& lbi : lbis) {
          lbi_groups->back().emplace_back(lbi);
          cur_group_model_size += MemSize4Lbi(lbi);
          if (cur_group_model_size >= GetCurGroupSize()) {
            lbi_groups->emplace_back(std::vector<LogicalBlobId>{});
            cur_group_model_size = 0;
          }
        }
      });
}

void GroupAllReducedLbis(
    const Job& job, const OpGraph& op_graph,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    std::vector<std::vector<LogicalBlobId>>* lbi_groups) {
  std::vector<LogicalBlobId> lbis;
  FindAllReducedLbis(job, op_graph, ProducerOpNode4Lbi, &lbis);
  SortAllReducedLbis(op_graph, ProducerOpNode4Lbi, &lbis);
  GroupAllReducedLbisByStrategy(ProducerOpNode4Lbi, lbis, lbi_groups);
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
