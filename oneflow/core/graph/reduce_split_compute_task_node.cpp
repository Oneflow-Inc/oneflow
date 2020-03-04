#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/graph/reduce_split_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

void ReduceSplitCompTaskNode::ProduceAllRegstsAndBindEdges() {
  HashMap<LogicalBlobId, int32_t> lbi2order;
  std::shared_ptr<Operator> reduce_split_op = this->logical_node()->SoleOp();
  FOR_RANGE(int32_t, idx, 0, reduce_split_op->output_bns().size()) {
    ProduceRegst("out_" + std::to_string(idx), false, 1, 1);
    const auto& lbi = reduce_split_op->BnInOp2Lbi(reduce_split_op->output_bns().Get(idx));
    CHECK(lbi2order.emplace(lbi, idx).second);
  }

  ForEachOutDataEdge([&](TaskEdge* edge) {
    TaskNode* dst_node = edge->dst_node();
    CHECK(edge->dst_node()->GetTaskType() == TaskType::kOptimizer
          || edge->dst_node()->GetTaskType() == TaskType::kNormalForward);
    CompTaskNode* mdupdt_node = dynamic_cast<CompTaskNode*>(dst_node);
    std::shared_ptr<Operator> mdupdt_op = mdupdt_node->logical_node()->SoleOp();
    for (const std::string& ibn : mdupdt_op->input_bns()) {
      const auto& order_it = lbi2order.find(mdupdt_op->BnInOp2Lbi(ibn));
      if (order_it != lbi2order.end()) {
        BindEdgeWithProducedRegst(edge, "out_" + std::to_string(order_it->second));
      }
    }
  });
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
    std::string blob_name = "out_" + std::to_string(i);
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(blob_name);
    CHECK(out_regst.get() != nullptr);
    out_regst->AddLbi(reduce_split_op->BnInOp2Lbi(blob_name));
    node->BindBnWithRegst(blob_name, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void ReduceSplitCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  CHECK_EQ(GetRankCtx().TotalSegmentCount(), 1);
  std::shared_ptr<Operator> reduce_split_op = this->logical_node()->SoleOp();
  int64_t offset = 0;
  for (int i = 0; i < reduce_split_op->output_bns().size(); ++i) {
    RegstDesc* out_regst = GetProducedRegst("out_" + std::to_string(i)).get();
    ctx.EnableMemSharing4Regst(out_regst, offset);
    offset += InferRegstSize(*out_regst);
  }
}

void ReduceSplitCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
