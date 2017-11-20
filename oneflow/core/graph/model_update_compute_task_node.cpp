#include "oneflow/core/graph/model_update_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void MdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto model_regst = ProduceRegst("model", 1, kMaxRegisterNum);
  auto model_tmp_regst = ProduceRegst("model_tmp", 1, 1);
  for (TaskEdge* out_edge : out_edges()) {
    TaskNode* dst_node = out_edge->dst_node();
    if (dst_node->GetTaskType() == TaskType::kForward
        || dst_node->GetTaskType() == TaskType::kBackward) {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("model_tmp", model_tmp_regst);
    } else if (dst_node->GetTaskType() == TaskType::kMdSave) {
      out_edge->AddRegst("model", model_regst);
    } else {
      UNEXPECTED_RUN();
    }
  }
}

void MdUpdtCompTaskNode::ConsumeAllRegsts() {
  if (JobDesc::Singleton()->IsPredict()) return;
  ConsumeRegst("model_diff_acc", SoleInEdge()->GetSoleRegst());
}

bool MdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("model")->IsLocked()
         && GetProducedRegst("model_tmp")->IsLocked();
}

void MdUpdtCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  std::shared_ptr<RegstDesc> model_diff_acc_regst = nullptr;
  BlobDesc packed_blob_desc;
  if (JobDesc::Singleton()->IsTrain()) {
    model_diff_acc_regst = SoleInEdge()->GetSoleRegst();
    packed_blob_desc = model_diff_acc_regst->CompPackedBlobDesc();
  } else {
    packed_blob_desc =
        BlobDesc(Shape(), JobDesc::Singleton()->DefaultDataType(), false);
  }
  node->BindBnInOpAndRegst("model_diff_acc", model_diff_acc_regst);
  auto model_regst = GetProducedRegst("model");
  node->BindBnInOpAndRegst(node->op()->SoleObn(), model_regst);
  auto data_tmp_regst = ProduceRegst("data_tmp", 1, 1);
  for (const std::string& dtbn : node->op()->data_tmp_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
    data_tmp_regst->AddLbn(lbn);
    node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
  node->op()->InferBlobDescs(
      [&](const std::string& bn_in_op) -> BlobDesc* {
        if (bn_in_op == "model_diff_acc") { return &packed_blob_desc; }
        return node->GetBlobDesc4BnInOpFunc()(bn_in_op);
      },
      nullptr);
}

void MdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  task_proto->set_random_seed(random_seed_);
}

}  // namespace oneflow
