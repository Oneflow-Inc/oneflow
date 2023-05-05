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
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"

namespace oneflow {

namespace {

const OpNode* OpNodeOnEdge(TaskEdge* edge, TaskNode* (TaskEdge::*GetNode)() const,
                           void (TaskNode::*ForEachDataEdge)(const std::function<void(TaskEdge*)>&)
                               const) {
  CompTaskNode* target_node = nullptr;
  do {
    TaskNode* tmp_node = (edge->*GetNode)();
    target_node = dynamic_cast<CompTaskNode*>(tmp_node);
    edge = nullptr;
    (tmp_node->*ForEachDataEdge)([&](TaskEdge* e) {
      if (edge == nullptr) { edge = e; }
    });
  } while (!target_node && edge);
  if (target_node) { return target_node->op_node(); }
  return nullptr;
}

std::vector<CompTaskNode*> GetCompTaskNodesOnEdge(
    TaskEdge* edge, TaskNode* (TaskEdge::*GetNode)() const,
    void (TaskNode::*ForEachDataEdge)(const std::function<void(TaskEdge*)>&) const) {
  std::queue<TaskNode*> nodes;
  HashSet<TaskNode*> visited_nodes;
  nodes.push((edge->*GetNode)());
  CHECK(visited_nodes.emplace((edge->*GetNode)()).second);
  std::vector<CompTaskNode*> comp_task_nodes;
  while (!nodes.empty()) {
    TaskNode* node = nodes.front();
    nodes.pop();
    CompTaskNode* comp_task_node = dynamic_cast<CompTaskNode*>(node);
    if (comp_task_node) {
      comp_task_nodes.emplace_back(comp_task_node);
    } else {
      (node->*ForEachDataEdge)([&](TaskEdge* task_edge) {
        if (visited_nodes.find((task_edge->*GetNode)()) == visited_nodes.end()) {
          nodes.push((task_edge->*GetNode)());
          CHECK(visited_nodes.emplace((task_edge->*GetNode)()).second);
        }
      });
    }
  }
  return comp_task_nodes;
}

std::shared_ptr<RegstDesc> NewFakeDataRegstDesc() {
  auto regst_desc = std::make_shared<RegstDesc>();
  regst_desc->mut_regst_desc_type()->mutable_data_regst_desc();
  return regst_desc;
}

}  // namespace

void CompTaskNode::ConsumeFakeRegst(const std::string& regst_name) {
  ConsumeRegst(regst_name, NewFakeDataRegstDesc());
  fake_consumed_regst_names_.insert(regst_name);
}

void CompTaskNode::ConsumeFakeRegstsIf() {
  ConsumeFakeRegsts();
  RegstDesc* data_regst_desc = nullptr;
  for (const auto& pair : consumed_regsts()) {
    for (const auto& regst_desc : pair.second) {
      if (regst_desc->regst_desc_type().has_data_regst_desc()) {
        // Only one fake data regst is creatd for each CompTaskNode with ConsumeFakeRegsts().
        CHECK(data_regst_desc == nullptr);
        data_regst_desc = CHECK_NOTNULL(regst_desc.get());
      } else if (regst_desc->regst_desc_type().has_ctrl_regst_desc()) {
        // do nothing.
      } else {
        UNIMPLEMENTED();
      }
    }
  }
  if (data_regst_desc != nullptr) {
    for (const auto& ibn : op_node()->op().input_bns()) {
      // Only one fake data regst is creatd and just use it for all input_bns as a placeholder.
      data_regst_desc->AddLbi(op_node()->op().BnInOp2Lbi(ibn));
    }
  }
}

void CompTaskNode::EraseFakeRegstsIf() {
  for (const auto& fake_consumed_regst_name : fake_consumed_regst_names_) {
    EraseConsumedRegstsByName(fake_consumed_regst_name);
  }
  fake_consumed_regst_names_.clear();
}

std::string CompTaskNode::VisualStr() const { return op_node_->op().op_name(); }

void CompTaskNode::InitFromProtoExceptConsumedRegsts(const TaskProto& proto) {
  TaskNode::InitFromProtoExceptConsumedRegsts(proto);
  parallel_ctx_ = proto.parallel_ctx();
}

void CompTaskNode::ToProto(TaskProto* task_proto, bool check) const {
  TaskNode::ToProto(task_proto, check);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
}

const OpNode* CompTaskNode::GetOneSuccOpNodeOnEdge(TaskEdge* edge) {
  return OpNodeOnEdge(edge, &TaskEdge::dst_node, &TaskNode::ForEachOutDataEdge);
}

const OpNode* CompTaskNode::GetOnePredOpNodeOnEdge(TaskEdge* edge) {
  return OpNodeOnEdge(edge, &TaskEdge::src_node, &TaskNode::ForEachInDataEdge);
}

std::vector<CompTaskNode*> CompTaskNode::GetSuccCompTaskNodesOnEdge(TaskEdge* edge) const {
  return GetCompTaskNodesOnEdge(edge, &TaskEdge::dst_node, &TaskNode::ForEachOutDataEdge);
}

std::vector<CompTaskNode*> CompTaskNode::GetPredCompTaskNodesOnEdge(TaskEdge* edge) const {
  return GetCompTaskNodesOnEdge(edge, &TaskEdge::src_node, &TaskNode::ForEachInDataEdge);
}

void CompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> op_time_shape(new Shape(*CHECK_JUST(op()->GetOpTimeShape())));
  ForEachProducedDataRegst([op_time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = op_time_shape;
  });
}

CompTaskNode* NewCompTaskNode4OpNode(const OpNode* op_node) {
  const OperatorConf& op_conf = op_node->op().op_conf();
  if (op_conf.has_user_conf()) {
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (IsClassRegistered<std::string, OpCompTaskNodeCreator>(op_type_name)) {
      return std::unique_ptr<OpCompTaskNodeCreator>(
                 NewObj<std::string, OpCompTaskNodeCreator>(op_type_name))
          ->NewCompTaskNode(op_conf);
    } else {
      return new NormalForwardCompTaskNode;
    }
  } else {
    OperatorConf::OpTypeCase op_type_case = op_conf.op_type_case();
    if (IsClassRegistered<int32_t, OpCompTaskNodeCreator>(op_type_case)) {
      return std::unique_ptr<OpCompTaskNodeCreator>(
                 NewObj<int32_t, OpCompTaskNodeCreator>(op_type_case))
          ->NewCompTaskNode(op_conf);
    } else {
      return new NormalForwardCompTaskNode;
    }
  }
}

}  // namespace oneflow
