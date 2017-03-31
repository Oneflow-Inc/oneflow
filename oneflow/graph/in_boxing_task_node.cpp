#include "graph/in_boxing_task_node.h"
#include <algorithm>
#include "operator/operator_factory.h"
#include "graph/comp_task_node.h"

namespace oneflow {

namespace {

const CompTaskNode* GetSuccCompTaskNode(const TaskEdge* edge) {
  const TaskNode* node = edge->dst_node();
  const CompTaskNode* ret = nullptr;
  while (ret = dynamic_cast<const CompTaskNode*>(node), ret == nullptr) {
    node = node->SoleOutEdge()->dst_node();
  }
  return ret;
}

}

void InBoxingTaskNode::FwBuildExecGraphAndSetProducedRegisterDescs() {
  SetOutEdgeRegisterPtr();
  Chain2EdgesMap chain2sorted_in_edges;
  FwInitChain2SortedEdgesMaps(&chain2sorted_in_edges, 
                              &TaskNode::in_edges,
                              &TaskEdge::src_node,
                              &TaskNode::SoleInEdge);
  std::vector<const TaskEdge*> sorted_out_edges;
  FwInitSortedOutEdges(&sorted_out_edges);
  for (const ChainEdgesPair& chain_sorted_in_edges : chain2sorted_in_edges) {
    FwBuildChainSortedEdgesPair(chain_sorted_in_edges, sorted_out_edges);
  }
  SetProducedRegister();
  mut_exec_graph().UpdateSourceAndSink();
}

void InBoxingTaskNode::FwInitSortedOutEdges(
    std::vector<const TaskEdge*>* sorted_out_edges) {
  sorted_out_edges->assign(out_edges().begin(), out_edges().end());
  std::sort(sorted_out_edges->begin(), sorted_out_edges->end(), []
      (const TaskEdge* lhs, const TaskEdge* rhs) {
    const CompTaskNode* lhs_node = GetSuccCompTaskNode(lhs);
    const CompTaskNode* rhs_node = GetSuccCompTaskNode(rhs);
    return lhs_node->parallel_id() < rhs_node->parallel_id();
  });
}

void InBoxingTaskNode::FwBuildChainSortedEdgesPair(
    const ChainEdgesPair& chain_sorted_in_edges,
    const std::vector<const TaskEdge*>& sorted_out_edges) {
  
  const ChainNode* in_chain = chain_sorted_in_edges.first;
  const auto& sorted_in_edges = chain_sorted_in_edges.second;

  ParallelDesc::Policy in_policy = in_chain->parallel_desc()->policy();
  ParallelDesc::Policy out_policy = chain_node()->parallel_desc()->policy();
  OpPair (*FwBuildBoxingOp)();
  if (in_policy == kDataParallel && out_policy == kDataParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpDataData;
  } else if (in_policy == kDataParallel && out_policy == kModelParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpDataModel;
  } else if (in_policy == kModelParallel && out_policy == kDataParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpModelData;
  } else if (in_policy == kModelParallel && out_policy == kModelParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpModelModel;
  }

  std::vector<std::string> lbns = FindLbnsBetween(in_chain, chain_node());
  for (const std::string& lbn : lbns) {
    OpPair op_pair = FwBuildBoxingOp();
    // First Node
    ExecNode* first_node = mut_exec_graph().NewExecNode();
    first_node->mut_op() = op_pair.first;
    for (const TaskEdge* edge : sorted_in_edges) {
      first_node->AddConsumedLbnRegiPair(lbn, GetRelatedRegister(edge));
    }
    // Second Node
    ExecNode* second_node = mut_exec_graph().NewExecNode();
    second_node->mut_op() = op_pair.second;
    for (const TaskEdge* edge : sorted_out_edges) {
      second_node->AddProducedLbnRegiPair(lbn, GetRelatedRegister(edge));
    }
    // Connect
    Connect(first_node, mut_exec_graph().NewExecEdge(lbn), second_node);
  }
}

void InBoxingTaskNode::SetProducedRegister() {
  AddInPathLbn2ProducedRegister();
  std::unique_ptr<RegisterDesc> boxing_middle_register(new DisContigRegistDesc);
  for (const std::unique_ptr<ExecEdge>& edge : exec_graph().edges()) {
    boxing_middle_register->AddPbn(edge->pbn());
  }
  AddProducedRegisterDesc("boxing_middle", std::move(boxing_middle_register));
}

namespace {

inline RegisterDesc* GetBpRegisterFromFwRegister(RegisterDesc* fw_register) {
  const TaskEdge* fw_edge = GetRelatedTaskEdge(fw_register);
  const TaskEdge* bp_edge = fw_edge->related_fwbp_edge();
  return GetRelatedRegister(bp_edge);
}

}

void InBoxingTaskNode::BpBuildExecGraphAndSetProducedRegisterDescs() {
  SetOutEdgeRegisterPtr();
  const ExecGraph& fw_exec_graph = GetFwNode()->exec_graph();
  std::unordered_map<const ExecNode*, ExecNode*> fw_node2bp_node;
  for (const std::unique_ptr<ExecNode>& fw_node: fw_exec_graph.nodes()) {
    ExecNode* bp_node = mut_exec_graph().NewExecNode();
    CHECK(fw_node2bp_node.emplace(fw_node.get(), bp_node).second);
    bp_node->mut_op() = fw_node->op();
    for (const auto& fw_pair : fw_node->consumed_lbn_regi_pairs()) {
      RegisterDesc* bp_register = GetBpRegisterFromFwRegister(fw_pair.second);
      bp_node->AddProducedLbnRegiPair(fw_pair.first, bp_register);
    }
    for (const auto& fw_pair : fw_node->produced_lbn_regi_pairs()) {
      RegisterDesc* bp_register = GetBpRegisterFromFwRegister(fw_pair.second);
      bp_node->AddConsumedLbnRegiPair(fw_pair.first, bp_register);
    }
  }
  for (const std::unique_ptr<ExecEdge>& fw_edge : fw_exec_graph.edges()) {
    Connect(fw_node2bp_node.at(fw_edge->dst_node()),
            mut_exec_graph().NewExecEdge(fw_edge->lbn()),
            fw_node2bp_node.at(fw_edge->src_node()));
  }
  mut_exec_graph().UpdateSourceAndSink();
  SetProducedRegister();
}

} // namespace oneflow
