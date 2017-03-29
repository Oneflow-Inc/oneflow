#include "graph/boxing_task_node.h"
#include "operator/operator_factory.h"

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
  mut_exec_graph().UpdateSourceAndSink();
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

namespace {

std::shared_ptr<const Operator> FwBuildBoxingOpDataData(
    const std::string& lbn,
    bool is_fw_in_boxing) {
  OperatorConf op_conf;
  op_conf.set_name("");
  BoxingOpConf* boxing_op_conf = op_conf.mutable_boxing_op_conf();
  boxing_op_conf->set_lbn(lbn);
  ConcatBoxConf* concat_in_conf = boxing_op_conf->mutable_concat_in_box_conf();
  SplitBoxConf* split_out_conf = boxing_op_conf->mutable_split_out_box_conf();
  concat_in_conf->set_axis(0);
  concat_in_conf->clear_proportion();
  split_out_conf->set_axis(0);
  split_out_conf->clear_proportion();
  return ConstructOpFromPbConf(op_conf);
}

std::shared_ptr<const Operator> FwBuildBoxingOpDataModel(
    const std::string& lbn,
    bool is_fw_in_boxing) {
  LOG(FATAL) << "TODO";
}
std::shared_ptr<const Operator> FwBuildBoxingOpModelData(
    const std::string& lbn,
    bool is_fw_in_boxing) {
  LOG(FATAL) << "TODO";
}
std::shared_ptr<const Operator> FwBuildBoxingOpModelModel(
    const std::string& lbn,
    bool is_fw_in_boxing) {
  LOG(FATAL) << "TODO";
}

}

void BoxingTaskNode::FwBuildChainPair(const Chain2EdgesPair& in_pair,
                                      const Chain2EdgesPair& out_pair) {
  using Policy = ParallelDesc::Policy;
  using std::placeholders::_1;
  using std::placeholders::_2;
  
  const ChainNode* in_chain = in_pair.first;
  const std::vector<const TaskEdge*>& in_edges = in_pair.second;
  const ChainNode* out_chain = out_pair.first;
  const std::vector<const TaskEdge*>& out_edges = out_pair.second;

  Policy in_policy = in_chain->parallel_desc()->policy();
  Policy out_policy = out_chain->parallel_desc()->policy();
  std::shared_ptr<const Operator> (*FwBuildBoxingOp)(const std::string&, bool);

  if (in_policy == kDataParallel && out_policy == kDataParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpDataData;
  } else if (in_policy == kDataParallel && out_policy == kModelParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpDataModel;
  } else if (in_policy == kModelParallel && out_policy == kDataParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpDataModel;
  } else if (in_policy == kModelParallel && out_policy == kModelParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpDataModel;
  }

  std::vector<std::string> lbns = FindLbnsBetween(in_chain, out_chain);
  for (const std::string& lbn : lbns) {
    ExecNode* boxing_node = mut_exec_graph().NewExecNode();
    std::shared_ptr<const Operator> op = FwBuildBoxingOp(lbn, IsFwInBoxing());
    boxing_node->mut_op() = op;
    for (const TaskEdge* edge : in_edges) {
      boxing_node->AddConsumedLbnRegiPair(lbn, edge->register_desc());
    }
    for (const TaskEdge* edge : out_edges) {
      boxing_node->AddProducedLbnRegiPair(lbn, edge->register_desc());
    }
  }
}

} // namespace oneflow
