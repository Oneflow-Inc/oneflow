#include "oneflow/core/graph/model_diff_accumulate_compute_task_node.h"
#include "model_diff_accumulate_compute_task_node.h"

namespace oneflow {

void MdDiffAccCompTaskNode::FixPackedBlobDescOfProducedRegst() {
  std::shared_ptr<RegstDesc> md_diff_acc_regst = GetProducedRegst("acc");
  CHECK(md_diff_acc_regst->IsLocked());
  Shape& shape = md_diff_acc_regst->MutBlobDesc(GenPackedLbi())->mut_shape();
  shape = Shape({static_cast<int64_t>(RoundUp(shape.elem_cnt(), parallel_ctx()->parallel_num()))});
}

void MdDiffAccCompTaskNode::InferProducedRegstTimeShape() {
  std::shared_ptr<Shape> time_shape;
  time_shape.reset(new Shape({Global<JobDesc>::Get()->TotalBatchNum()}));
  for (auto& pair : produced_regsts()) { pair.second->mut_time_shape() = time_shape; }
}

}  // namespace oneflow
