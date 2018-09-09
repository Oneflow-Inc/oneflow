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

}  // namespace oneflow
