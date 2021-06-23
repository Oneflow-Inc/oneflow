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
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

class SspVariableProxyCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SspVariableProxyCompTaskNode);
  SspVariableProxyCompTaskNode() = default;
  ~SspVariableProxyCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override {
    int64_t buffer_size = user_op::UserOpConfWrapper(op()->op_conf()).attr<int64_t>("buffer_size");
    CHECK_GT(buffer_size, 0);
    ProduceRegst("value", false, buffer_size, buffer_size);
    ProduceRegst("ref", false, 1, 1);
    HashMap<std::string, std::set<TaskEdge*>> out_regst_name2edges;
    ForEachOutDataEdge(
        [&](TaskEdge* edge) {
          {
            auto* copy_hd_node = dynamic_cast<CopyHdTaskNode*>(edge->dst_node());
            if (copy_hd_node != nullptr) {
              // The only possible regst_name is "value" because "ref" is always strictly one-to-one
              // connected.
              CHECK_EQ(*out_regst_name2edges["value"].insert(edge).first, edge);
              return;
            }
          }
          auto* dst_node = dynamic_cast<CompTaskNode*>(edge->dst_node());
          CHECK(dst_node != nullptr)
              << "SspVariableProxyTaskNode must be consumed by CompTaskNode. got "
              << TaskType_Name(edge->dst_node()->GetTaskType());
          for (const std::string& ibn : dst_node->op()->input_bns()) {
            const LogicalBlobId& dst_in_lbi = dst_node->op()->BnInOp2Lbi(ibn);
            if (dst_in_lbi == op()->BnInOp2Lbi("ref_0")) {
              CHECK_EQ(*out_regst_name2edges["ref"].insert(edge).first, edge);
            } else if (dst_in_lbi == op()->BnInOp2Lbi("value_0")) {
              CHECK_EQ(*out_regst_name2edges["value"].insert(edge).first, edge);
            } else {
              // do nothing
            }
          }
        });
    for (const auto& pair : out_regst_name2edges) {
      for (TaskEdge* edge : pair.second) { BindEdgeWithProducedRegst(edge, pair.first); }
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
    exec_node->mut_op() = op();
    exec_node->BindBnWithOneOfTheRegsts("var_0", GetConsumedRegst("var"));
    BindInplacebetweenVarAndRef();
  }

  void BindInplacebetweenVarAndRef() {
    const auto& var_regst = GetSoleConsumedRegst("var");
    CHECK_EQ(var_regst->NumOfLbi(), 1);
    CHECK_EQ(var_regst->min_register_num(), 1);
    CHECK_EQ(var_regst->max_register_num(), 1);
    const auto& ref_regst = GetProducedRegst("ref");
    ref_regst->set_force_inplace_consumed_regst_desc_id(var_regst->regst_desc_id());
  }

  void BuildOutRegst() {
    ExecNode* exec_node = mut_exec_gph().SoleNode();
    const auto& AddLbiAndBindBn = [&](const std::string& regst_name) {
      // "ref_0" obn <-> "ref" regst_name
      // "value_0" obn <-> "value" regst_name
      const std::string& obn = regst_name + "_0";
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

}  // namespace oneflow
