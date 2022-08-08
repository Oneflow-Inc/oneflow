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
#include <iterator>
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/graph/task_stream_index_manager.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/register/register_desc.h"

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

std::string GetOutRegstNameByObn(const std::string& obn) { return "__" + obn; }

}  // namespace

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
    if (op_type_name == "identity_buffer") {
      mem_block_num = user_op::UserOpConfWrapper(sole_op->op_conf()).attr<int64_t>("buffer_size");
    }
  }
  // when output blob num > 1 and task node on out edge is all NormalForwardCompTaskNode ,
  // create multi out regst by output blob name in op

  HashMap<LogicalBlobId, std::string> lbi2out_regst_name;
  for (const std::string& obn : sole_op->output_bns()) {
    const LogicalBlobId& lbi = sole_op->BnInOp2Lbi(obn);
    std::string out_regst_name = GetOutRegstNameByObn(obn);
    lbi2out_regst_name.insert({lbi, out_regst_name});
    ProduceOutRegstByNameAndBlockNum(out_regst_name, mem_block_num);
  }
  ForEachOutDataEdge([&](TaskEdge* edge) {
    for (const LogicalBlobId& lbi : edge->GetLbis()) {
      auto it = lbi2out_regst_name.find(lbi);
      CHECK(it != lbi2out_regst_name.end());
      BindEdgeWithProducedRegst(edge, it->second);
    }
  });
  ProduceRegst("tmp", true);
}

void NormalForwardCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) {
    for (const auto& regst : edge->GetRegsts()) { ConsumeRegst("in", regst); }
  });
}

void NormalForwardCompTaskNode::HandleInplaceOperationRegsts() {
  const auto& _op = op();
  if (_op->op_conf().has_user_conf()) {
    const auto& inplace_operation_info = _op->op_conf().user_conf().inplace_operation_info();

    for (const auto& it : inplace_operation_info) {
      const std::string& obn = it.first;
      const auto& input_arg_index_pair = it.second;

      const std::string out_regst_name = GetOutRegstNameByObn(obn);
      std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_regst_name);

      const std::string& input_lbn = _op->op_conf()
                                         .user_conf()
                                         .input()
                                         .at(input_arg_index_pair.arg())
                                         .s(input_arg_index_pair.index());
      const LogicalBlobId input_lbi = GenLogicalBlobId(input_lbn);

      std::shared_ptr<RegstDesc> in_regst = nullptr;
      for (TaskEdge* in_edge : in_edges()) {
        CompTaskNode* comp_task_node = dynamic_cast<CompTaskNode*>(in_edge->src_node());
        if (!comp_task_node) continue;
        if (comp_task_node->op()->op_name() == input_lbi.op_name()) {
          in_regst = comp_task_node->GetProducedRegst(GetOutRegstNameByObn(input_lbi.blob_name()));
          break;
        }
      }

      CHECK(in_regst != nullptr) << "Must have found in_regst at this point! But operation: "
                                 << _op->op_name() << " of obn: " << obn
                                 << " does not have an associated in_regst!"
                                 << " The asscociated in_regst is with lbn: " << input_lbi.op_name()
                                 << "/" << input_lbi.blob_name();

      // set in_regst's memory can be reused
      // let out_regst reuse in_regst's memory when the operation is executed
      in_regst->set_enable_reuse_mem(true);
      out_regst->set_hint_inplace_consumed_regst_desc_id(in_regst->regst_desc_id());
    }
  }
}

void NormalForwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  BuildTmp7BufRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void NormalForwardCompTaskNode::BuildExecGphStructAndBindInRegst() {
  ExecNode* cur_node = mut_exec_gph().NewNode();
  cur_node->mut_op() = op();
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  for (const std::string& ibn : cur_node->op()->input_bns()) {
    cur_node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
  }
}

void NormalForwardCompTaskNode::BuildOutRegst() {
  ExecNode* exec_node = mut_exec_gph().SoleNode();
  for (const std::string& obn : exec_node->op()->output_bns()) {
    std::string out_regst_name = GetOutRegstNameByObn(obn);
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_regst_name);
    out_regst->AddLbi(exec_node->op()->BnInOp2Lbi(obn));
    exec_node->BindBnWithRegst(obn, out_regst);
  }
}

void NormalForwardCompTaskNode::BuildTmp7BufRegsts() {
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  });
}

REGISTER_COMP_TASK_STREAM_INDEX_GETTER(TaskType::kNormalForward);

}  // namespace oneflow
