#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
namespace oneflow {

void NormalMdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int32_t min_model_regst = -1;
  int32_t max_model_regst = -1;
  if (JobDesc::Singleton()->IsPredict()) {
    min_model_regst = 1;
    max_model_regst = 1;
  } else if (JobDesc::Singleton()->Staleness() == -1) {
    min_model_regst = 2;
    max_model_regst = kMaxRegisterNum;
  } else {
    min_model_regst = 1;
    max_model_regst = JobDesc::Singleton()->Staleness() + 1;
  }
  auto model_regst = ProduceRegst("model", min_model_regst, max_model_regst);
  auto model_tmp_regst = ProduceRegst("model_tmp", 1, 1);
  ProduceRegst("data_tmp", 1, 1);
  bool found_related_init_model_task = false;
  for (TaskEdge* out_edge : out_edges()) {
    TaskNode* dst_node = out_edge->dst_node();
    if (IsForwardTaskType(dst_node->GetTaskType())
        || IsBackwardTaskType(dst_node->GetTaskType())) {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("model_tmp", model_tmp_regst);
      if (IsForwardTaskType(dst_node->GetTaskType())
          && !found_related_init_model_task) {
        auto fw_node = static_cast<ForwardCompTaskNode*>(dst_node);
        fw_node->set_random_seed(random_seed_);
        found_related_init_model_task = true;
      }
    } else {
      out_edge->AddRegst("model", model_regst);
    }
  }
}

void NormalMdUpdtCompTaskNode::ConsumeAllRegsts() {
  if (JobDesc::Singleton()->IsPredict()) { return; }
  for (TaskEdge* edge : in_edges()) {
    ConsumeRegst("model_diff_acc_" + NewUniqueId(), edge->GetSoleRegst());
  }
}

bool NormalMdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("model")->IsLocked()
         && GetProducedRegst("model_tmp")->IsLocked();
}

static std::shared_ptr<const Operator> ConstructModelUpdateOp(int32_t in_num) {
  OperatorConf op_conf;
  op_conf.set_name("md_update_" + NewUniqueId());
  NormalModelUpdateOpConf* mdupdt_conf = op_conf.mutable_normal_mdupdt_conf();
  const JobDesc* job_desc = JobDesc::Singleton();
  if (job_desc->IsTrain()) {
    *(mdupdt_conf->mutable_user_conf()) =
        job_desc->job_conf().train_conf().model_update_conf();
  }
  mdupdt_conf->set_in_num(in_num);
  return ConstructOp(op_conf);
}

void NormalMdUpdtCompTaskNode::BuildExecGphAndRegst() {
  if (JobDesc::Singleton()->IsPredict()) { return; }
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ConstructModelUpdateOp(consumed_regsts().size());
  size_t ibn_idx = 0;
  for (const auto& pair : consumed_regsts()) {
    node->BindBnInOpAndRegst(node->op()->input_bns().at(ibn_idx++),
                             pair.second.lock());
  }
  node->BindBnInOpAndRegst(node->op()->SoleObn(), GetProducedRegst("model"));
  auto data_tmp_regst = GetProducedRegst("data_tmp");
  for (const std::string& dtbn : node->op()->data_tmp_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
    data_tmp_regst->AddLbn(lbn);
    node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
  node->InferBlobDescs(nullptr, device_type());
}

void NormalMdUpdtCompTaskNode::LockRegsts() {
  GetProducedRegst("data_tmp")->Lock();
}

void NormalMdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  ForEachNodeOnOutEdge([&](const TaskNode* node) {
    if (IsForwardTaskType(node->GetTaskType())) {
      if (task_proto->related_init_model_task_id() == -1) {
        task_proto->set_related_init_model_task_id(node->task_id());
      }
    } else if (IsBackwardTaskType(node->GetTaskType())) {
      // do nothing
    } else {
      CHECK_EQ(task_proto->related_save_model_task_id(), -1);
      task_proto->set_related_save_model_task_id(node->task_id());
    }
  });
}

}  // namespace oneflow
