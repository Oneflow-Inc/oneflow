/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

namespace {

size_t RegstNum4OpSameOutputBlob(OperatorConf::OpTypeCase op_type_case) {
  if (IsClassRegistered<int32_t, RuntimeRegstNum4OpSameOutputBlob>(op_type_case)) {
    std::unique_ptr<RuntimeRegstNum4OpSameOutputBlob> ptr;
    ptr.reset(NewObj<int32_t, RuntimeRegstNum4OpSameOutputBlob>(op_type_case));
    return *ptr;
  } else {
    return -1;
  }
}

std::string GetOutRegstNameByObn(const std::string& obn) {
  return "NormalForwardCompTaskNodeOutRegstName_" + obn;
}

}  // namespace

bool NormalForwardCompTaskNode::HasBackwardCompTaskNode() { return false; }

bool NormalForwardCompTaskNode::CanProduceSeperatedRegstsForEachOutBlob() const {
  return op()->output_bns().size() > 1 && IsAllOutNodeNormalForward();
}

bool NormalForwardCompTaskNode::IsAllOutNodeNormalForward() const {
  bool ret = true;
  ForEachNodeOnOutDataEdge([&](TaskNode* node) {
    auto* fw_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
    if (fw_node == nullptr) { ret = false; }
  });
  return ret;
}

void NormalForwardCompTaskNode::ProduceOutRegstByNameAndBlockNum(const std::string& name,
                                                                 size_t mem_block_num) {
  if (mem_block_num != -1) {
    CHECK_GT(mem_block_num, 0);
    ProduceRegst(name, false, mem_block_num, mem_block_num);
  } else {
    ProduceRegst(name, true);
  }
}

void NormalForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<const Operator> sole_op = op();
  size_t mem_block_num = RegstNum4OpSameOutputBlob(sole_op->op_conf().op_type_case());
  if (sole_op->op_conf().has_user_conf()) {
    const std::string& op_type_name = sole_op->op_conf().user_conf().op_type_name();
    const auto* op_reg_result = user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
    CHECK(op_reg_result != nullptr) << "op_type_name " << op_type_name << " not register";
    if (op_reg_result->same_output_regst_num > 0) {
      mem_block_num = op_reg_result->same_output_regst_num;
    }
  }
  // when output blob num > 1 and task node on out edge is all NormalForwardCompTaskNode ,
  // create multi out regst by output blob name in op
  if (CanProduceSeperatedRegstsForEachOutBlob()) {
    HashMap<LogicalBlobId, std::string> lbi2out_regst_name;
    for (const std::string& obn : sole_op->output_bns()) {
      const LogicalBlobId& lbi = sole_op->BnInOp2Lbi(obn);
      std::string out_regst_name = GetOutRegstNameByObn(obn);
      lbi2out_regst_name.insert({lbi, out_regst_name});
      ProduceOutRegstByNameAndBlockNum(out_regst_name, mem_block_num);
    }
    ForEachOutDataEdge([&](TaskEdge* edge) {
      TaskNode* node = edge->dst_node();
      auto* dst_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
      CHECK(dst_node != nullptr) << "1regst1blob ONLY support normal fw comp task node 121";
      std::shared_ptr<const Operator> dst_op = dst_node->op();
      bool is_found = false;
      for (const std::string& ibn : dst_op->input_bns()) {
        const LogicalBlobId& dst_in_lbi = dst_op->BnInOp2Lbi(ibn);
        if (lbi2out_regst_name.find(dst_in_lbi) != lbi2out_regst_name.end()) {
          is_found = true;
          BindEdgeWithProducedRegst(edge, lbi2out_regst_name.at(dst_in_lbi));
        }
      }
      CHECK(is_found) << "Cannot find comsumed blob in dst op: " << dst_op->op_name();
    });
  } else {
    ProduceOutRegstByNameAndBlockNum("out", mem_block_num);
    ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
  }
  ProduceRegst("tmp", true);
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
  ExecNode* cur_node = mut_exec_gph().NewNode();
  cur_node->mut_op() = op();
  for (const std::string& obn : op()->output_bns()) {
    const LogicalBlobId& lbi = op()->BnInOp2Lbi(obn);
    CHECK(lbi2producer.insert({lbi, {cur_node, obn}}).second);
  }
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
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
}

void NormalForwardCompTaskNode::BuildOutRegst() {
  if (CanProduceSeperatedRegstsForEachOutBlob()) {
    ExecNode* exec_node = mut_exec_gph().SoleNode();
    for (const std::string& obn : exec_node->op()->output_bns()) {
      std::string out_regst_name = GetOutRegstNameByObn(obn);
      std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_regst_name);
      out_regst->AddLbi(exec_node->op()->BnInOp2Lbi(obn));
      exec_node->BindBnWithRegst(obn, out_regst);
    }
  } else {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
    ExecNode* exec_node = mut_exec_gph().SoleNode();
    for (const std::string& obn : exec_node->op()->output_bns()) {
      out_regst->AddLbi(exec_node->op()->BnInOp2Lbi(obn));
      exec_node->BindBnWithRegst(obn, out_regst);
    }
  }
}

void NormalForwardCompTaskNode::BuildTmp7BufRegsts() {
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  });
}

}  // namespace oneflow
