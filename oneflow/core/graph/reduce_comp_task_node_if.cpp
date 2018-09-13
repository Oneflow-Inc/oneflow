#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"
#include "task_node.h"

namespace oneflow {

int64_t InferRegstSize(const RegstDesc& regst) {
  return RtBlobDesc(*(regst.GetBlobDesc(GenPackedLbi()))).ByteSizeOfDataContentField();
}

void BuildCtrlRegstBetweenReduceCopyNodes(const CompTaskNode* src_reduce,
                                          const CompTaskNode* dst_reduce, int64_t copy_node_num) {
  struct ReduceCopyNodePair {
    TaskNode* copy_h2d;
    TaskNode* copy_d2h;
    ReduceCopyNodePair() : copy_h2d(nullptr), copy_d2h(nullptr) {}
  };
  HashMap<int64_t, ReduceCopyNodePair> mem_shared_offset2copy_nodes;

  for (TaskEdge* out_edge : src_reduce->out_edges()) {
    if (out_edge->dst_node()->GetTaskType() == TaskType::kCopyHd) {
      int64_t offset = out_edge->GetSoleRegst()->mem_shared_offset();
      mem_shared_offset2copy_nodes[offset].copy_d2h = out_edge->dst_node();
    }
  }
  CHECK_EQ(copy_node_num, mem_shared_offset2copy_nodes.size());

  for (TaskEdge* in_edge : dst_reduce->in_edges()) {
    if (in_edge->src_node()->GetTaskType() == TaskType::kCopyHd) {
      int64_t offset = in_edge->GetSoleRegst()->mem_shared_offset();
      CHECK(mem_shared_offset2copy_nodes.find(offset) != mem_shared_offset2copy_nodes.end());
      mem_shared_offset2copy_nodes.at(offset).copy_h2d = in_edge->src_node();
    }
  }

  for (const auto& kv : mem_shared_offset2copy_nodes) {
    kv.second.copy_d2h->BuildCtrlRegstDesc(kv.second.copy_h2d);
  }
}

void BuildCtrlRegstBetweenReduceAddAndGather(CompTaskNode* src_reduce,
                                             const CompTaskNode* dst_reduce,
                                             int64_t copy_node_num) {
  HashMap<int64_t, int64_t> mem_shared_offset2copy_regst_desc_id;
  for (TaskEdge* in_edge : src_reduce->in_edges()) {
    if (in_edge->src_node()->GetTaskType() == TaskType::kCopyHd) {
      std::shared_ptr<RegstDesc> in_regst = in_edge->GetSoleRegst();
      int64_t offset = in_regst->mem_shared_offset();
      CHECK(mem_shared_offset2copy_regst_desc_id.emplace(offset, in_regst->regst_desc_id()).second);
    }
  }
  CHECK_EQ(copy_node_num, mem_shared_offset2copy_regst_desc_id.size());
  for (TaskEdge* in_edge : dst_reduce->in_edges()) {
    if (in_edge->src_node()->GetTaskType() == TaskType::kCopyHd) {
      int64_t offset = in_edge->GetSoleRegst()->mem_shared_offset();
      auto copy_it = mem_shared_offset2copy_regst_desc_id.find(offset);
      CHECK(copy_it != mem_shared_offset2copy_regst_desc_id.end());
      RegstDesc* ctrl_regst_desc = src_reduce->BuildCtrlRegstDesc(in_edge->src_node());
      RegstDescTypeProto* ctrl_regst_desc_type_proto = ctrl_regst_desc->mut_regst_desc_type();
      CHECK(ctrl_regst_desc_type_proto->has_ctrl_regst_desc());
      ctrl_regst_desc_type_proto->mutable_ctrl_regst_desc()->set_reliant_regst_desc_id(
          copy_it->second);
    }
  }
}

}  // namespace oneflow
