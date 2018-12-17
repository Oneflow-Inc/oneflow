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
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    CHECK(dst_node->GetTaskType() == TaskType::kNormalMdUpdt);
    CompTaskNode* mdupdt_node = dynamic_cast<CompTaskNode*>(dst_node);
    for (TaskEdge* mdupdt_edge : mdupdt_node->out_edges()) {
      if (IsBackwardTaskType(mdupdt_edge->dst_node()->GetTaskType())) {
        CompTaskNode* bw_node = dynamic_cast<CompTaskNode*>(mdupdt_edge->dst_node());
        // There may be multiple out_regsts on the same edge for shared_model app
        EdgeInfo edge_info{edge, bw_node->order_in_graph()};
        edge_infos.emplace_back(edge_info);
      }
    }
  }
  SortEdges(&edge_infos);
  FOR_RANGE(size_t, idx, 0, edge_infos.size()) {
    std::string out_regst_name = "out_" + std::to_string(idx);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name, false, 1, 1);
    edge_infos[idx].edge->AddRegst(out_regst_name, out_regst);
  }
}

void ReduceSplitCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", this->SoleInEdge()->GetSoleRegst());
}

void ReduceSplitCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_split_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_split_op;
  node->BindBnWithRegst(reduce_split_op->SoleIbn(), GetSoleConsumedRegst("in"));

  TaskNode* reduce_concat_node = FindPredReduceTaskNodeIf(
      [](TaskNode* node) { return node->GetTaskType() == TaskType::kReduceConcat; });
  CHECK(reduce_concat_node);

  CHECK_EQ(reduce_concat_node->consumed_regsts().size(), GetDataRegstDescCnt(produced_regsts()));
  FOR_RANGE(size_t, i, 0, reduce_split_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    CHECK(out_regst.get() != nullptr);
    out_regst->CopyBlobDescFrom(
        reduce_concat_node->GetSoleConsumedRegst("in_" + std::to_string(i)).get());
    node->BindBnWithRegst(reduce_split_op->output_bns().Get(i), out_regst);
  }
}

void ReduceSplitCompTaskNode::FixPackedBlobDescOfProducedRegst() {
  int64_t out_regst_num = GetDataRegstDescCnt(produced_regsts());
  FOR_RANGE(int64_t, idx, 0, out_regst_num) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(idx));
    CHECK(out_regst->IsLocked());
    Shape& shape = out_regst->MutBlobDesc(GenPackedLbi())->mut_shape();
    shape =
        Shape({static_cast<int64_t>(RoundUp(shape.elem_cnt(), parallel_ctx()->parallel_num()))});
  }
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
