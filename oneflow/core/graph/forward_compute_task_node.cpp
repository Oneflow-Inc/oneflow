#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void ForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out");
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (SuccChainNodeOnEdge(edge) == chain_node()) {
      VirtualAddRegstOnRecurrentOutEdge(edge);
    } else {
      edge->AddRegst("out", out_regst);
      if (IsBackwardTaskType(dst_node->GetTaskType())) {
        edge->AddRegst("activation", ProduceRegst("activation"));
        edge->AddRegst("data_tmp", ProduceRegst("data_tmp"));
      }
    }
  }
}

void ForwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (src_node->GetTaskType() == TaskType::kNormalMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else if (src_node->GetTaskType() == TaskType::kNormalizationMdUpdt) {
      ConsumeRegst("other_model", edge->GetSoleRegst());
    } else {
      VirtualConsumeInRegst(edge);
    }
  }
}

void ForwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  BuildActivationRegst();
  BuildModelAndTmpRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) {
    node->InferBlobDescs(parallel_ctx(), device_type());
  });
}

void ForwardCompTaskNode::LockRegsts() {
  TaskNode::LockRegsts();
  TryLockConsumedRegst("model");
  TryLockConsumedRegst("model_tmp");
  TryLockConsumedRegst("other_model");
}

void ForwardCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  task_proto->set_random_seed(random_seed_);
}

void ForwardCompTaskNode::VirtualAddRegstOnRecurrentOutEdge(TaskEdge* edge) {
  UNIMPLEMENTED();
}

void ForwardCompTaskNode::BuildActivationRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetProducedRegst("activation");
  mut_exec_gph().ForEachEdge([&](const ExecEdge* edge) {
    if (activation_regst->GetBlobDesc(edge->lbn()) == nullptr) {
      activation_regst->AddLbn(edge->lbn());
      edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
    }
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
  });
}

void ForwardCompTaskNode::BuildModelAndTmpRegsts() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  std::shared_ptr<RegstDesc> model_regst = GetConsumedRegst("model");
  std::shared_ptr<RegstDesc> model_tmp_regst = GetConsumedRegst("model_tmp");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
      data_tmp_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      if (!model_tmp_regst->IsLocked()) {
        const std::string& lbn = node->op()->Lbn4BnInOp(mtbn);
        model_tmp_regst->AddLbn(lbn);
      }
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mbn : node->op()->model_bns()) {
      if (!model_regst->IsLocked()) {
        const std::string& lbn = node->op()->Lbn4BnInOp(mbn);
        model_regst->AddLbn(lbn);
      }
      node->BindBnInOpAndRegst(mbn, model_regst);
    }
    if (node->op()->IsNormalizationOp()) {
      std::shared_ptr<RegstDesc> other_model_regst =
          GetConsumedRegst("other_model");
      for (const std::string& otbn : node->op()->other_bns()) {
        const std::string& lbn = node->op()->Lbn4BnInOp(otbn);
        other_model_regst->AddLbn(lbn);
        node->BindBnInOpAndRegst(otbn, other_model_regst);
      }
    }
  });
}

}  // namespace oneflow
