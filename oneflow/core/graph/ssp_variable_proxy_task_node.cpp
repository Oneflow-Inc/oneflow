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
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

class SspVariableProxyCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SspVariableProxyCompTaskNode);
  SspVariableProxyCompTaskNode() = default;
  ~SspVariableProxyCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override {
    const auto& op = *logical_node()->SoleOp();
    int64_t buffer_size = user_op::UserOpConfWrapper(op.op_conf()).attr<int64_t>("buffer_size");
    CHECK_GT(buffer_size, 0);
    ProduceRegst("value", false, buffer_size, buffer_size);
    ProduceRegst("ref", false, 1, 1);
    HashMap<std::string, TaskEdge*> out_regst_name2edge;
    ForEachOutDataEdge([&](TaskEdge* edge) {
      auto* dst_node = dynamic_cast<CompTaskNode*>(edge->dst_node());
      CHECK(dst_node != nullptr) << "SspVariableProxyTaskNode consumed by non CompTaskNode";
      const Operator& dst_op = *dst_node->logical_node()->SoleOp();
      for (const std::string& ibn : dst_op.input_bns()) {
        const LogicalBlobId& dst_in_lbi = dst_op.BnInOp2Lbi(ibn);
        if (dst_in_lbi == op.BnInOp2Lbi("ref")) {
          CHECK_EQ(out_regst_name2edge.emplace("ref", edge).first->second, edge);
        } else if (dst_in_lbi == op.BnInOp2Lbi("value")) {
          CHECK_EQ(out_regst_name2edge.emplace("value", edge).first->second, edge);
        } else {
          // do nothing
        }
      }
    });
    for (const auto& pair : out_regst_name2edge) {
      BindEdgeWithProducedRegst(pair.second, pair.first);
    }
  }
  void ConsumeAllRegsts() override {
    ConsumeRegst("var");
    ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("var", edge->GetSoleRegst()); });
  }

  TaskType GetTaskType() const override { return TaskType::kSspVariableProxy; }

 private:
  void BuildExecGphAndRegst() override {
    BuildExecGphStructAndBindInRegst();
    BuildOutRegst();
    mut_exec_gph().TopoForEachNode(
        [this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
  }

  void BuildExecGphStructAndBindInRegst() {
    ExecNode* exec_node = mut_exec_gph().NewNode();
    exec_node->mut_op() = logical_node()->SoleOp();
    exec_node->BindBnWithOneOfTheRegsts("var", GetConsumedRegst("var"));
  }

  void BuildOutRegst() {
    ExecNode* exec_node = mut_exec_gph().SoleNode();
    const auto& AddLbiAndBindBn = [&](const std::string& obn) {
      // "ref" obn <-> "ref" regst_name
      // "value" obn <-> "value" regst_name
      const std::string& regst_name = obn;
      const std::shared_ptr<RegstDesc>& regst = GetProducedRegst(regst_name);
      regst->AddLbi(exec_node->op()->BnInOp2Lbi(obn));
      exec_node->BindBnWithRegst(obn, regst);
    };
    AddLbiAndBindBn("ref");
    AddLbiAndBindBn("value");
  }

  void InferProducedDataRegstTimeShape() override { NaiveInferProducedDataRegstTimeShape(); }
};

REGISTER_USER_OP_COMP_TASK_NODE_TYPE("ssp_variable_proxy", SspVariableProxyCompTaskNode);
REGISTER_USER_OP_INDEPENDENT_AREA_ID("ssp_variable_proxy");

}  // namespace oneflow
