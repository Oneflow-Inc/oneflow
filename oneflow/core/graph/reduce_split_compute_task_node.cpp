#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/graph/reduce_split_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

namespace {

int32_t GetDataRegstDescCnt(
    const HashMap<std::string, std::shared_ptr<RegstDesc>> name2regst_desc) {
  size_t cnt = 0;
  for (const auto& pair : name2regst_desc) {
    cnt += pair.second->regst_desc_type().has_data_regst_desc();
  }
  return cnt;
}

}  // namespace

void ReduceSplitCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::vector<EdgeInfo> edge_infos;
  std::shared_ptr<Operator> reduce_split_op = this->logical_node()->SoleOp();
  HashMap<LogicalBlobId, int32_t> lbi2order;
  FOR_RANGE(int32_t, idx, 0, reduce_split_op->output_bns().size()) {
    const auto& lbi = reduce_split_op->BnInOp2Lbi(reduce_split_op->output_bns().Get(idx));
    CHECK(lbi2order.emplace(lbi, idx).second);
  }
  ForEachOutDataEdge([&](TaskEdge* edge) {
    TaskNode* dst_node = edge->dst_node();
    CHECK(edge->dst_node()->GetTaskType() == TaskType::kOptimizer
          || edge->dst_node()->GetTaskType() == TaskType::kNormalForward);
    CompTaskNode* mdupdt_node = dynamic_cast<CompTaskNode*>(dst_node);
    std::shared_ptr<Operator> mdupdt_op = mdupdt_node->logical_node()->SoleOp();
    int32_t order = -1;
    for (const std::string& ibn : mdupdt_op->input_bns()) {
      const auto& order_it = lbi2order.find(mdupdt_op->BnInOp2Lbi(ibn));
      if (order_it != lbi2order.end()) { order = order_it->second; }
    }
    CHECK_NE(order, -1);
    EdgeInfo edge_info{edge, order};
    edge_infos.emplace_back(edge_info);
  });
  SortEdges(&edge_infos);
  FOR_RANGE(size_t, idx, 0, edge_infos.size()) {
    std::string out_regst_name = "out_" + std::to_string(idx);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name, false, 1, 1);
    edge_infos[idx].edge->AddRegst(out_regst_name, out_regst);
  }
}

void ReduceSplitCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", this->SoleInDataEdge()->GetSoleRegst());
}

TaskNode* ReduceSplitCompTaskNode::GetPrevReduceTaskNode(TaskType task_type) {
  CHECK(task_type == TaskType::kReduceConcat || task_type == TaskType::kReduceIdentity);
  TaskNode* task_node =
      FindPredReduceTaskNodeIf([&](TaskNode* node) { return node->GetTaskType() == task_type; });
  CHECK_NOTNULL(task_node);
  return task_node;
}

void ReduceSplitCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_split_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_split_op;
  node->BindBnWithRegst(reduce_split_op->SoleIbn(), GetSoleConsumedRegst("in"));

  FOR_RANGE(size_t, i, 0, reduce_split_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    CHECK(out_regst.get() != nullptr);
    out_regst->AddLbi(reduce_split_op->BnInOp2Lbi(reduce_split_op->output_bns().Get(i)));
    node->BindBnWithRegst(reduce_split_op->output_bns().Get(i), out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void ReduceSplitCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  CHECK_EQ(GetRankCtx().TotalSegmentCount(), 1);
  size_t split_num = GetDataRegstDescCnt(produced_regsts());
  int64_t offset = 0;
  FOR_RANGE(int32_t, idx, 0, split_num) {
    RegstDesc* split_out_regst = GetProducedRegst("out_" + std::to_string(idx)).get();
    ctx.EnableMemSharing4Regst(split_out_regst, offset);
    offset += InferRegstSize(*split_out_regst);
  }
}

void ReduceSplitCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
