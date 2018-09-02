#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NormalForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", true);
  ProduceRegst("fw_pb_out", false);
  ProduceRegst("activation", true);
  ProduceRegst("data_tmp", true);
  ProduceRegst("fw_buf", true, 1, 1);
  ProduceRegst("forward_model", false);
  ProduceRegst("const_buf", false, 1, 1);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() == "MdSave") {
      BindEdgeWithProducedRegst(edge, "forward_model");
    } else if (succ_logical->TypeName() == "NormalBackward") {
      BindEdgeWithProducedRegst(edge, "out");
      BindEdgeWithProducedRegst(edge, "activation");
      BindEdgeWithProducedRegst(edge, "data_tmp");
      BindEdgeWithProducedRegst(edge, "const_buf");
    } else {
      BindEdgeWithProducedRegst(edge, "out");
      if (edge->dst_node()->GetTaskType() != TaskType::kCopyHd) {
        BindEdgeWithProducedRegst(edge, "fw_pb_out");
      }
    }
  }
}

void NormalForwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskType src_task_type = src_node->GetTaskType();
    if (src_task_type == TaskType::kNormalMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("const_model", edge->GetRegst("const_model"));
    } else {
      edge->ForEachRegst(
          [&](std::shared_ptr<RegstDesc> regst_desc) { ConsumeRegst("in", regst_desc); });
    }
  }
}

bool NormalForwardCompTaskNode::IsReadyForBuild() {
  for (std::shared_ptr<RegstDesc> regst_desc : GetConsumedRegst("in")) {
    if (regst_desc->IsLocked() == false) { return false; }
  }
  return true;
}

void NormalForwardCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  task_proto->set_random_seed(random_seed_);
}

void NormalForwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  BuildActivationRegst();
  BuildModel7ConstModel7DataTmp7BufRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void NormalForwardCompTaskNode::LockRegsts() {
  TaskNode::LockRegsts();
  TryLockConsumedRegst("model");
  TryLockConsumedRegst("const_model");
}

void NormalForwardCompTaskNode::BuildExecGphStructAndBindInRegst() {
  HashMap<LogicalBlobId, std::pair<ExecNode*, std::string>> lbi2producer;
  for (std::shared_ptr<const Operator> op : logical_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    op->ForEachOutputBn([&](const std::string& obn) {
      const LogicalBlobId& lbi = op->BnInOp2Lbi(obn);
      CHECK(lbi2producer.insert({lbi, {cur_node, obn}}).second);
    });
  }
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    cur_node->op()->ForEachInputBn([&](const std::string& ibn) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(ibn);
      auto producer_it = lbi2producer.find(lbi);
      if (producer_it != lbi2producer.end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbi(lbi);
        edge->mut_src_bn() = producer_it->second.second;
        edge->mut_dst_bn() = ibn;
        Connect(producer_it->second.first, edge, cur_node);
      } else {
        cur_node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
      }
    });
  });
}

void NormalForwardCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> pb_out_regst = GetProducedRegst("fw_pb_out");
  auto ForEachOutBn7Regst =
      [&](const ExecNode* cur_node,
          const std::function<void(const std::string&, std::shared_ptr<RegstDesc>)>& Handler) {
        for (const auto& obn : cur_node->op()->output_bns()) { Handler(obn, out_regst); }
        for (const auto& pobn : cur_node->op()->pb_output_bns()) { Handler(pobn, pb_out_regst); }
      };
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<LogicalBlobId> found_lbis;
    for (ExecEdge* out_edge : cur_node->out_edges()) { found_lbis.insert(out_edge->lbi()); }
    ForEachOutBn7Regst(cur_node, [&](const std::string& obn, std::shared_ptr<RegstDesc> out_regst) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(obn);
      if (logical_node()->IsDataLbiOnOutEdge(lbi)) {
        out_regst->AddLbi(lbi);
        cur_node->BindBnWithRegst(obn, out_regst);
      } else {
        CHECK(found_lbis.empty() || found_lbis.find(lbi) != found_lbis.end());
      }
    });
  });
}

void NormalForwardCompTaskNode::BuildActivationRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetProducedRegst("activation");
  mut_exec_gph().ForEachEdge([&](const ExecEdge* edge) {
    if (activation_regst->GetBlobDesc(edge->lbi()) == nullptr) {
      activation_regst->AddLbi(edge->lbi());
      edge->src_node()->BindBnWithRegst(edge->src_bn(), activation_regst);
    }
    edge->dst_node()->BindBnWithRegst(edge->dst_bn(), activation_regst);
  });
}

void NormalForwardCompTaskNode::BuildModel7ConstModel7DataTmp7BufRegsts() {
  std::shared_ptr<RegstDesc> model_regst = GetSoleConsumedRegst("model");
  std::shared_ptr<RegstDesc> const_model_regst = GetSoleConsumedRegst("const_model");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
    node->AddBnToRegstAndBindIt(&Operator::fw_buf_bns, GetProducedRegst("fw_buf"));
    node->AddBnToRegstAndBindIt(&Operator::const_buf_bns, GetProducedRegst("const_buf"));
    node->AddBnToRegstAndBindIt(&Operator::forward_model_bns, GetProducedRegst("forward_model"));
    for (const std::string& cmbn : node->op()->const_model_bns()) {
      if (!const_model_regst->IsLocked()) {
        const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(cmbn);
        const_model_regst->AddLbi(lbi);
      }
      node->BindBnWithRegst(cmbn, const_model_regst);
    }
    for (const std::string& mbn : node->op()->model_bns()) {
      if (!model_regst->IsLocked()) {
        const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(mbn);
        model_regst->AddLbi(lbi);
      }
      node->BindBnWithRegst(mbn, model_regst);
    }
  });
}

}  // namespace oneflow
