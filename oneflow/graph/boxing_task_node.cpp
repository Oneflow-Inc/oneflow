#include "graph/boxing_task_node.h"

namespace oneflow {

void BoxingTaskNode::SetFwInBoxing() {
  CHECK(IsFwNode());
  is_fw_in_boxing_ = true;
}

void BoxingTaskNode::SetFwOutBoxing() {
  CHECK(IsFwNode());
  is_fw_in_boxing_ = false;
}

void BoxingTaskNode::InitWithFwNode(TaskNode* fw_node) {
  TaskNode::InitWithFwNode(fw_node);
  is_fw_in_boxing_ =
      of_dynamic_cast<BoxingTaskNode*>(fw_node)->is_fw_in_boxing_;
}

void BoxingTaskNode::FwBuildExecGraphAndSetProducedRegisterDescs() {
  FwSetOutEdgeRegisterPtr();
  Chain2EdgesMap chain2in_edges;
  Chain2EdgesMap chain2out_edges;
  FwInitChain2EdgesMaps(&chain2in_edges, &chain2out_edges);
  for (const auto& in_pair : chain2in_edges) {
    for (const auto& out_pair : chain2out_edges) {
      FwBuildChainPair(in_pair, out_pair);
    }
  }
}

void BoxingTaskNode::FwSetOutEdgeRegisterPtr() {
  for (TaskEdge* edge : out_edges()) {
    std::string name = "boxing_out_" + std::to_string(edge->edge_id());
    std::unique_ptr<RegisterDesc> register_desc(new DisContigRegistDesc);
    edge->set_register_desc(register_desc.get());
    AddProducedRegisterDesc(name, std::move(register_desc));
  }
}

void BoxingTaskNode::FwInitChain2EdgesMaps(Chain2EdgesMap* chain2in_edges,
                                           Chain2EdgesMap* chain2out_edges) {
  for (TaskEdge* edge : in_edges()) {
    const ChainNode* chain = edge->src_node()->chain_node();
    (*chain2in_edges)[chain].push_back(edge);
  }
  for (TaskEdge* edge : out_edges()) {
    const ChainNode* chain = edge->dst_node()->chain_node();
    (*chain2out_edges)[chain].push_back(edge);
  }
}

void BoxingTaskNode::FwBuildChainPair(const Chain2EdgesPair& in_pair,
                                      const Chain2EdgesPair& out_pair) {
  using Policy = ParallelDesc::Policy;

  const ChainNode* in_chain = in_pair.first;
  const std::vector<const TaskEdge*> in_edges = in_pair.second;
  const ChainNode* out_chain = out_pair.first;
  const std::vector<const TaskEdge*> out_edges = out_pair.second;

  Policy in_policy = in_chain->parallel_desc()->policy();
  Policy out_policy = out_chain->parallel_desc()->policy();

  std::vector<std::string> lbns = FindLbnsBetween(in_chain, out_chain);
  for (const std::string& lbn : lbns) {
    if (in_policy == kDataParallel && out_policy == kDataParallel) {
      FwBuildEdgesPairDataData(in_edges, out_edges, lbn);
    } else if (in_policy == kDataParallel && out_policy == kModelParallel) {
      FwBuildEdgesPairDataModel(in_edges, out_edges, lbn);
    } else if (in_policy == kModelParallel && out_policy == kDataParallel) {
      FwBuildEdgesPairModelData(in_edges, out_edges, lbn);
    } else if (in_policy == kModelParallel && out_policy == kModelParallel) {
      FwBuildEdgesPairModelModel(in_edges, out_edges, lbn);
    } else {
      LOG(FATAL) << "invalid";
    }
  }
}

void BoxingTaskNode::FwBuildEdgesPairDataData(
    const std::vector<const TaskEdge*>& in_edges,
    const std::vector<const TaskEdge*>& out_edges,
    const std::string& lbn) {
  // For now, we can not inference the correct prop of concat in a elegant way
  // do that in Setup step
  LOG(FATAL) << "TODO";
}

void BoxingTaskNode::FwBuildEdgesPairDataModel(
    const std::vector<const TaskEdge*>& in_edges,
    const std::vector<const TaskEdge*>& out_edges,
    const std::string& lbn) {
  LOG(FATAL) << "TODO";
}

void BoxingTaskNode::FwBuildEdgesPairModelData(
    const std::vector<const TaskEdge*>& in_edges,
    const std::vector<const TaskEdge*>& out_edges,
    const std::string& lbn) {
  LOG(FATAL) << "TODO";
}

void BoxingTaskNode::FwBuildEdgesPairModelModel(
    const std::vector<const TaskEdge*>& in_edges,
    const std::vector<const TaskEdge*>& out_edges,
    const std::string& lbn) {
  LOG(FATAL) << "TODO";
}

} // namespace oneflow
