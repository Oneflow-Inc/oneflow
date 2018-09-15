#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

const NormalForwardCompTaskNode* NormalMdUpdtCompTaskNode::GetForwardTaskNode() const {
  for (const TaskEdge* out_edge : out_edges()) {
    const TaskNode* dst_node = out_edge->dst_node();
    if (IsForwardTaskType(dst_node->GetTaskType())) {
      return dynamic_cast<const NormalForwardCompTaskNode*>(dst_node);
    }
  }
  UNIMPLEMENTED();
}

bool NormalMdUpdtCompTaskNode::IsTrainable() const {
  return Global<JobDesc>::Get()->IsTrain()
         && GetForwardTaskNode()->logical_node()->SoleOp()->op_conf().trainable();
}

void NormalMdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int32_t max_model_regst = 1;
  auto model_regst = ProduceRegst("model", false, 1, max_model_regst);
  auto const_model_regst = ProduceRegst("const_model", false, 1, 1);
  ProduceRegst("processed_model_diff", false, 1, 1);
  ProduceRegst("data_tmp", false, 1, 1);
  related_init_model_task_id_ = -1;
  for (TaskEdge* out_edge : out_edges()) {
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
      out_edge->AddRegst("model", model_regst);
    }
  }
}

void NormalMdUpdtCompTaskNode::ConsumeAllRegsts() {
  if (!IsTrainable()) { return; }
  for (TaskEdge* edge : in_edges()) {
    auto regst_descs = edge->GetRegsts();
    for (auto& regst_desc : regst_descs) {
      ConsumeRegst("model_diff_acc_" + NewUniqueId(), regst_desc);
    }
  }
}

bool NormalMdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("model")->IsLocked() && GetProducedRegst("const_model")->IsLocked();
}

void NormalMdUpdtCompTaskNode::BuildExecGphAndRegst() {
  if (!IsTrainable()) { return; }
  ExecNode* shared_model_diff_add_node = mut_exec_gph().NewNode();
  shared_model_diff_add_node->mut_op() = logical_node()->SoleOp();
  size_t ibn_idx = 0;
  for (const auto& pair : consumed_regsts()) {
    shared_model_diff_add_node->BindBnWithRegst(
        shared_model_diff_add_node->op()->input_bns().Get(ibn_idx++), pair.second.front());
  }
  std::shared_ptr<RegstDesc> processed_model_diff_regst = GetProducedRegst("processed_model_diff");
  shared_model_diff_add_node->BindBnWithRegst(logical_node()->SoleOp()->SoleObn(),
                                              processed_model_diff_regst);
  // "model" regst is already bound with lbis and locked by the corresponding
  // NormalForwardCompTaskNode
  processed_model_diff_regst->CopyBlobDescFrom(GetProducedRegst("model").get());

  ExecNode* model_update_node = nullptr;
  ExecEdge* exec_edge = nullptr;
  processed_model_diff_regst->ForEachLbi([&](const LogicalBlobId& lbi) {
    OperatorConf op_conf;
    op_conf.set_name("md_update_" + lbi.op_name() + "_" + lbi.blob_name());
    op_conf.set_device_type(logical_node()->parallel_desc()->device_type());
    op_conf.mutable_normal_mdupdt_conf()->set_model_diff(lbi.op_name() + '/' + lbi.blob_name());
    op_conf.mutable_normal_mdupdt_conf()->set_model(lbi.op_name() + '/' + lbi.blob_name());
    if (Global<JobDesc>::Get()->IsTrain()) {
      *(op_conf.mutable_normal_mdupdt_conf()->mutable_user_conf()) =
          Global<JobDesc>::Get()->other_conf().train_conf().model_update_conf();
    }
    std::shared_ptr<Operator> model_update_op = ConstructOp(op_conf);
    model_update_node = mut_exec_gph().NewNode();
    model_update_node->mut_op() = model_update_op;
    exec_edge = mut_exec_gph().NewEdge();
    exec_edge->set_lbi(lbi);
    exec_edge->mut_src_bn() = lbi.blob_name();
    exec_edge->mut_dst_bn() = model_update_op->SoleIbn();
    Connect(shared_model_diff_add_node, exec_edge, model_update_node);

    model_update_node->BindBnWithRegst(model_update_op->SoleIbn(), processed_model_diff_regst);
    model_update_node->BindBnWithRegst(model_update_op->SoleObn(), GetProducedRegst("model"));
    model_update_node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
  });
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void NormalMdUpdtCompTaskNode::LockRegsts() {
  GetProducedRegst("processed_model_diff")->Lock();
  GetProducedRegst("data_tmp")->Lock();
}

void NormalMdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  ForEachNodeOnOutEdge([&](const TaskNode* node) {
    if (IsForwardTaskType(node->GetTaskType())) {
      // do nothing
    } else if (IsBackwardTaskType(node->GetTaskType())) {
      // do nothing
    } else {
      CHECK_EQ(task_proto->related_save_model_task_id(), -1);
      task_proto->set_related_save_model_task_id(node->task_id());
    }
  });
  task_proto->set_related_init_model_task_id(related_init_model_task_id_);
}

}  // namespace oneflow
