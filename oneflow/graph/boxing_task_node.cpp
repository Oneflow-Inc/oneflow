#include "graph/boxing_task_node.h"
#include <algorithm>
#include "operator/operator_factory.h"

namespace oneflow {

auto BoxingTaskNode::FwBuildBoxingOpDataData() -> OpPair {
  OperatorConf first_op_conf;
  first_op_conf.set_name("");
  first_op_conf.mutable_concat_op_conf()->set_axis(0);
  OperatorConf second_op_conf;
  second_op_conf.set_name("");
  second_op_conf.mutable_split_op_conf()->set_axis(0);
  return {ConstructOpFromPbConf(first_op_conf),
          ConstructOpFromPbConf(second_op_conf)};
}

auto BoxingTaskNode::FwBuildBoxingOpDataModel() -> OpPair {
  OperatorConf first_op_conf;
  first_op_conf.set_name("");
  first_op_conf.mutable_concat_op_conf()->set_axis(0);
  OperatorConf second_op_conf;
  second_op_conf.set_name("");
  second_op_conf.mutable_clone_op_conf();
  return {ConstructOpFromPbConf(first_op_conf),
          ConstructOpFromPbConf(second_op_conf)};
}

auto BoxingTaskNode::FwBuildBoxingOpModelData() -> OpPair {
  OperatorConf first_op_conf;
  first_op_conf.set_name("");
  first_op_conf.mutable_concat_op_conf()->set_axis(1);
  OperatorConf second_op_conf;
  second_op_conf.set_name("");
  second_op_conf.mutable_split_op_conf()->set_axis(0);
  return {ConstructOpFromPbConf(first_op_conf),
          ConstructOpFromPbConf(second_op_conf)};
}

auto BoxingTaskNode::FwBuildBoxingOpModelModel() -> OpPair {
  OperatorConf first_op_conf;
  first_op_conf.set_name("");
  first_op_conf.mutable_concat_op_conf()->set_axis(1);
  OperatorConf second_op_conf;
  second_op_conf.set_name("");
  second_op_conf.mutable_clone_op_conf();
  return {ConstructOpFromPbConf(first_op_conf),
          ConstructOpFromPbConf(second_op_conf)};
}

void BoxingTaskNode::SetOutEdgeRegisterPtr() {
  for (TaskEdge* edge : out_edges()) {
    std::string name = "boxing_out_" + std::to_string(edge->edge_id());
    std::unique_ptr<RegisterDesc> register_desc(new DisContigRegistDesc);
    BindProducedRegisterAndOutEdge(register_desc.get(), edge);
    AddProducedRegisterDesc(name, std::move(register_desc));
  }
}

void BoxingTaskNode::FwInitChain2SortedEdgesMaps(
    Chain2EdgesMap* chain2sorted_edges,
    const std::unordered_set<TaskEdge*>& (TaskNode::*in_out_edges)() const,
    TaskNode* (TaskEdge::*src_dst_node)() const,
    TaskEdge* (TaskNode::*SoleEdge)() const) {

  chain2sorted_edges->clear();
  std::unordered_map<const TaskEdge*, const StageNode*> edge2stage;
  for (const TaskEdge* edge : (this->*in_out_edges)()) {
    const TaskNode* pred_succ_node = (edge->*src_dst_node)();
    while (pred_succ_node->chain_node() == chain_node()) {
      pred_succ_node = ((pred_succ_node->*SoleEdge)()->*src_dst_node)();
    }
    (*chain2sorted_edges)[pred_succ_node->chain_node()].push_back(edge);
    edge2stage[edge] = pred_succ_node->stage_node();
  }
  for (auto& pair : *chain2sorted_edges) {
    std::vector<const TaskEdge*>& edges = pair.second;
    std::sort(edges.begin(), edges.end(), [&edge2stage](const TaskEdge* lhs,
                                                        const TaskEdge* rhs) {
      const StageNode* lhs_stage = edge2stage.at(lhs);
      const StageNode* rhs_stage = edge2stage.at(rhs);
      CHECK(lhs_stage->chain_node() == rhs_stage->chain_node());
      return lhs_stage->parallel_range().begin <
             rhs_stage->parallel_range().begin;
    });
  }
}

} // namespace oneflow
