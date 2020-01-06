#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

namespace {

size_t RegstNum4OpSameOutputBlob(OperatorConf::OpTypeCase op_type_case) {
  if (IsClassRegistered<RuntimeRegstNum4OpSameOutputBlob>(op_type_case)) {
    std::unique_ptr<RuntimeRegstNum4OpSameOutputBlob> ptr;
    ptr.reset(NewObj<RuntimeRegstNum4OpSameOutputBlob>(op_type_case));
    return *ptr;
  } else {
    return -1;
  }
}

}  // namespace

bool NormalForwardCompTaskNode::HasBackwardCompTaskNode() { return false; }

void NormalForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  const Operator& op = *logical_node()->SoleOp();
  size_t mem_block_num = RegstNum4OpSameOutputBlob(op.op_conf().op_type_case());
  if (mem_block_num != -1) {
    ProduceRegst("out", false, mem_block_num, mem_block_num);
  } else {
    ProduceRegst("out", true);
  }
  ProduceRegst("tmp", true);
  ProduceRegst("const_buf", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void NormalForwardCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) {
    for (const auto& regst : edge->GetRegsts()) { ConsumeRegst("in", regst); }
  });
}

bool NormalForwardCompTaskNode::IsReadyForBuild() {
  for (std::shared_ptr<RegstDesc> regst_desc : GetConsumedRegst("in")) {
    if (regst_desc->IsLocked() == false) { return false; }
  }
  return true;
}

void NormalForwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  BuildTmp7BufRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void NormalForwardCompTaskNode::BuildExecGphStructAndBindInRegst() {
  HashMap<LogicalBlobId, std::pair<ExecNode*, std::string>> lbi2producer;
  for (std::shared_ptr<const Operator> op : logical_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_bns()) {
      const LogicalBlobId& lbi = op->BnInOp2Lbi(obn);
      CHECK(lbi2producer.insert({lbi, {cur_node, obn}}).second);
    }
  }
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
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
    }
  });
}

void NormalForwardCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<LogicalBlobId> found_lbis;
    for (ExecEdge* out_edge : cur_node->out_edges()) { found_lbis.insert(out_edge->lbi()); }
    for (const std::string& obn : cur_node->op()->output_bns()) {
      out_regst->AddLbi(cur_node->op()->BnInOp2Lbi(obn));
      cur_node->BindBnWithRegst(obn, out_regst);
    }
  });
}

void NormalForwardCompTaskNode::BuildTmp7BufRegsts() {
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
    node->AddBnToRegstAndBindIt(&Operator::const_buf_bns, GetProducedRegst("const_buf"));
  });
}

void NormalForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  CHECK(!in_regsts.empty());
  const std::shared_ptr<Shape>& in_time_shape = in_regsts.front()->data_regst_time_shape();
  for (const auto& regst : in_regsts) {
    CHECK(*in_time_shape == *(regst->data_regst_time_shape()));
  }

  ForEachProducedDataRegst([in_time_shape](const std::string& name, RegstDesc* regst) {
    if (name == "const_buf") {
      regst->mut_data_regst_time_shape()->reset(new Shape({1}));
    } else {
      *regst->mut_data_regst_time_shape() = in_time_shape;
    }
  });
}

}  // namespace oneflow
