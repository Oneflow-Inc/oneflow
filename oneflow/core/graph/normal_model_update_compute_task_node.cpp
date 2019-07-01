#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

const NormalForwardCompTaskNode* NormalMdUpdtCompTaskNode::GetForwardTaskNode() const {
  const NormalForwardCompTaskNode* ret = nullptr;
  ForEachOutDataEdge([&](TaskEdge* out_edge) {
    const TaskNode* dst_node = out_edge->dst_node();
    if (ret == nullptr && IsForwardTaskType(dst_node->GetTaskType())) {
      ret = dynamic_cast<const NormalForwardCompTaskNode*>(dst_node);
      CHECK_NOTNULL(ret);
    }
  });
  CHECK_NOTNULL(ret);
  return ret;
}

bool NormalMdUpdtCompTaskNode::IsTrainable() const {
  return Global<JobDesc>::Get()->IsTrain()
         && GetForwardTaskNode()->logical_node()->SoleOp()->op_conf().trainable();
}

void NormalMdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int32_t max_model_regst = 1;
  auto model_regst = ProduceRegst("model", false, 1, max_model_regst);
  auto const_model_regst = ProduceRegst("const_model", false, 1, 1);
  auto forward_model_regst = ProduceRegst("forward_model", false, 1, 1);
  ProduceRegst("processed_model_diff", false, 1, 1);
  ProduceRegst("data_tmp", false, 1, 1);
  related_init_model_task_id_ = -1;
  std::list<std::pair<std::string, std::shared_ptr<RegstDesc>>> model_to_save{
      {"model", model_regst}, {"forward_model", forward_model_regst}};
  ForEachOutDataEdge([&](TaskEdge* out_edge) {
    TaskNode* dst_node = out_edge->dst_node();
    if (IsForwardTaskType(dst_node->GetTaskType()) || IsBackwardTaskType(dst_node->GetTaskType())) {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("const_model", const_model_regst);
      if (IsForwardTaskType(dst_node->GetTaskType()) && related_init_model_task_id_ == -1) {
        auto fw_node = static_cast<NormalForwardCompTaskNode*>(dst_node);
        fw_node->set_random_seed(random_seed_);
        related_init_model_task_id_ = fw_node->task_id();
      }
    } else {
      out_edge->AddRegst(model_to_save.front().first, model_to_save.front().second);
      model_to_save.pop_front();
    }
  });
}

bool NormalMdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("model")->IsLocked() && GetProducedRegst("const_model")->IsLocked();
}

void NormalMdUpdtCompTaskNode::LockRegsts() {
  GetProducedRegst("processed_model_diff")->Lock();
  GetProducedRegst("data_tmp")->Lock();
  GetProducedRegst("forward_model")->Lock();
}

void NormalMdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  ForEachNodeOnOutEdge([&](const TaskNode* node) {
    if (IsForwardTaskType(node->GetTaskType())) {
      // do nothing
    } else if (IsBackwardTaskType(node->GetTaskType())) {
      // do nothing
    } else {
      task_proto->add_related_save_model_task_ids(node->task_id());
    }
  });
  task_proto->set_related_init_model_task_id(related_init_model_task_id_);
}

void NormalMdUpdtCompTaskNode::FixPackedBlobDescOfProducedRegst() {
  std::shared_ptr<RegstDesc> diff_add_out_regst = GetProducedRegst("processed_model_diff");
  CHECK(diff_add_out_regst->IsLocked());
  Shape& shape = diff_add_out_regst->MutBlobDesc(GenPackedLbi())->mut_shape();
  shape = Shape({static_cast<int64_t>(RoundUp(shape.elem_cnt(), parallel_ctx()->parallel_num()))});
}

void NormalMdUpdtCompTaskNode::InferProducedDataRegstTimeShape() {
  ForEachProducedDataRegst([](const std::string& name, RegstDesc* regst) {
    if (name == "const_model") {
      regst->mut_data_regst_time_shape()->reset(new Shape({1}));
    } else {
      regst->mut_data_regst_time_shape()->reset(
          new Shape({Global<JobDesc>::Get()->TotalBatchNum()}));
    }
  });
}

void NormalMdUpdtCompTaskNode::EnableMemSharingBetweenFirstInAndProcessedMdDiffRegst() {
  if (!IsTrainable()) { return; }
  ExecNode* diff_add_node = exec_gph().SoleSourceNode();
  RegstDesc* first_in_regst =
      diff_add_node->RegstDesc4BnInOp(diff_add_node->op()->input_bns().Get(0));
  RegstDesc* diff_add_out_regst = diff_add_node->RegstDesc4BnInOp(diff_add_node->op()->SoleObn());
  CHECK_EQ(diff_add_out_regst, GetProducedRegst("processed_model_diff").get());
  CHECK(first_in_regst->HasSameMemSize(diff_add_out_regst));
  if (!first_in_regst->HasSetMemSharedId()) {
    int64_t mem_shared_id = Global<IDMgr>::Get()->NewMemSharedId();
    first_in_regst->set_enable_mem_sharing(true);
    first_in_regst->set_mem_shared_id(mem_shared_id);
    first_in_regst->set_mem_shared_offset(0);
  }
  diff_add_out_regst->CopyMemSharedInfoFrom(first_in_regst);
}

}  // namespace oneflow
