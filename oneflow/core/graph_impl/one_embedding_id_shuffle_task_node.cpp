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
#include "oneflow/core/graph/one_embedding_id_shuffle_task_node.h"
#include "oneflow/core/graph/task_stream_index_manager.h"
#include "oneflow/core/framework/framework.h"

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

void OneEmbeddingIdShuffleTaskNode::ProduceOutRegstByNameAndBlockNum(const std::string& name,
                                                                     size_t mem_block_num) {
  if (mem_block_num != -1) {
    CHECK_GT(mem_block_num, 0);
    ProduceRegst(name, false, mem_block_num, mem_block_num);
  } else {
    ProduceRegst(name, true);
  }
}

void OneEmbeddingIdShuffleTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<const Operator> sole_op = op();
  size_t mem_block_num = RegstNum4OpSameOutputBlob(sole_op->op_conf().op_type_case());
  if (sole_op->op_conf().has_user_conf()) {
    const std::string& op_type_name = sole_op->op_conf().user_conf().op_type_name();
    const auto* op_reg_result = user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
    CHECK(op_reg_result != nullptr) << "op_type_name " << op_type_name << " not register";
    if (ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ENABLE_OVERLAP", true)) {
      mem_block_num = op_reg_result->same_output_regst_num;
    } else {
      mem_block_num = 1;
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

void OneEmbeddingIdShuffleTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) {
    for (const auto& regst : edge->GetRegsts()) { ConsumeRegst("in", regst); }
  });
}

void OneEmbeddingIdShuffleTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  BuildTmp7BufRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void OneEmbeddingIdShuffleTaskNode::BuildExecGphStructAndBindInRegst() {
  ExecNode* cur_node = mut_exec_gph().NewNode();
  cur_node->mut_op() = op();
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  for (const std::string& ibn : cur_node->op()->input_bns()) {
    cur_node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
  }
}

void OneEmbeddingIdShuffleTaskNode::BuildOutRegst() {
  ExecNode* exec_node = mut_exec_gph().SoleNode();
  for (const std::string& obn : exec_node->op()->output_bns()) {
    std::string out_regst_name = GetOutRegstNameByObn(obn);
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_regst_name);
    out_regst->AddLbi(exec_node->op()->BnInOp2Lbi(obn));
    exec_node->BindBnWithRegst(obn, out_regst);
  }
}

void OneEmbeddingIdShuffleTaskNode::BuildTmp7BufRegsts() {
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  });
}

REGISTER_COMP_TASK_STREAM_INDEX_GETTER(TaskType::kIdShuffle);

REGISTER_USER_OP_COMP_TASK_NODE_TYPE("id_shuffle", OneEmbeddingIdShuffleTaskNode);

}  // namespace oneflow
