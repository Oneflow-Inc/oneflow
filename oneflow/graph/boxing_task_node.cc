#include "graph/boxing_task_node.h"
#include <algorithm>
#include "operator/operator_factory.h"
#include "graph/comp_task_node.h"

namespace oneflow {

namespace {

using OpPair =
  std::pair<std::shared_ptr<const Operator>, std::shared_ptr<const Operator>>;

OpPair FwBuildBoxingOpDataData(size_t in_num, size_t out_num) {
  OperatorConf first_op_conf;
  first_op_conf.set_name("");
  first_op_conf.mutable_concat_op_conf()->set_axis(0);
  first_op_conf.mutable_concat_op_conf()->set_in_num(in_num);
  OperatorConf second_op_conf;
  second_op_conf.set_name("");
  second_op_conf.mutable_split_op_conf()->set_axis(0);
  second_op_conf.mutable_split_op_conf()->set_out_num(out_num);
  return {ConstructOpFromPbConf(first_op_conf),
          ConstructOpFromPbConf(second_op_conf)};
}

OpPair FwBuildBoxingOpDataModel(size_t in_num, size_t out_num) {
  OperatorConf first_op_conf;
  first_op_conf.set_name("");
  first_op_conf.mutable_concat_op_conf()->set_axis(0);
  first_op_conf.mutable_concat_op_conf()->set_in_num(in_num);
  OperatorConf second_op_conf;
  second_op_conf.set_name("");
  second_op_conf.mutable_clone_op_conf()->set_out_num(out_num);
  return {ConstructOpFromPbConf(first_op_conf),
          ConstructOpFromPbConf(second_op_conf)};
}

OpPair FwBuildBoxingOpModelData(size_t in_num, size_t out_num) {
  OperatorConf first_op_conf;
  first_op_conf.set_name("");
  first_op_conf.mutable_concat_op_conf()->set_axis(1);
  first_op_conf.mutable_concat_op_conf()->set_in_num(in_num);
  OperatorConf second_op_conf;
  second_op_conf.set_name("");
  second_op_conf.mutable_split_op_conf()->set_axis(0);
  second_op_conf.mutable_split_op_conf()->set_out_num(out_num);
  return {ConstructOpFromPbConf(first_op_conf),
          ConstructOpFromPbConf(second_op_conf)};
}

OpPair FwBuildBoxingOpModelModel(size_t in_num, size_t out_num) {
  OperatorConf first_op_conf;
  first_op_conf.set_name("");
  first_op_conf.mutable_concat_op_conf()->set_axis(1);
  first_op_conf.mutable_concat_op_conf()->set_in_num(in_num);
  OperatorConf second_op_conf;
  second_op_conf.set_name("");
  second_op_conf.mutable_clone_op_conf()->set_out_num(out_num);
  return {ConstructOpFromPbConf(first_op_conf),
          ConstructOpFromPbConf(second_op_conf)};
}

}

void BoxingTaskNode::FwBuildExecAndProducedRegsts(Path* path) {
  BindOutEdgeAndRegst();
  FwBuildExecGraph();
  SetProducedRegst();
}

void BoxingTaskNode::BindOutEdgeAndRegst() {
  for (TaskEdge* edge : out_edges()) {
    std::string name = "boxing_out_" + std::to_string(edge->edge_id());
    std::unique_ptr<RegstDesc> regst_desc(new DisContigRegstDesc);
    BindProducedRegstAndOutEdge(regst_desc.get(), edge);
    AddProducedRegstDesc(name, std::move(regst_desc));
  }
}

void BoxingTaskNode::FwInitChain2SortedEdgesMaps(
    Chain2EdgesMap* chain2sorted_edges,
    const std::unordered_set<TaskEdge*>& (TaskNode::*in_out_edges)() const,
    TaskNode* (TaskEdge::*src_dst_node)() const,
    TaskEdge* (TaskNode::*SoleEdge)() const) {

  chain2sorted_edges->clear();
  HashMap<const TaskEdge*, const StageNode*> edge2stage;
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
      return lhs_stage->parallel_range().begin() <
             rhs_stage->parallel_range().begin();
    });
  }
}

void BoxingTaskNode::FwSortEdgesInnerStage(
    std::vector<const TaskEdge*>* edges_to_be_sorted,
    TaskNode* (TaskEdge::*src_dst_node)() const,
    TaskEdge* (TaskNode::*SoleEdge)() const) {
  
  auto GetPredSuccCompTaskNode = [&](const TaskEdge* edge) {
    const TaskNode* node = (edge->*src_dst_node)();
    const CompTaskNode* ret = nullptr;
    while (ret = dynamic_cast<const CompTaskNode*>(node), ret == nullptr) {
      node = ((node->*SoleEdge)()->*src_dst_node)();
    }
    return ret;
  };
  std::sort(edges_to_be_sorted->begin(), edges_to_be_sorted->end(), [&]
      (const TaskEdge* lhs, const TaskEdge* rhs) {
    const CompTaskNode* lhs_node = GetPredSuccCompTaskNode(lhs);
    const CompTaskNode* rhs_node = GetPredSuccCompTaskNode(rhs);
    return lhs_node->parallel_id() < rhs_node->parallel_id();
  });
}

void BoxingTaskNode::FwBuildChainSortedEdgesPair(
    const ChainEdgesPair& chain_sorted_in_edges,
    const ChainEdgesPair& chain_sorted_out_edges) {
  
  const ChainNode* in_chain = chain_sorted_in_edges.first;
  const auto& sorted_in_edges = chain_sorted_in_edges.second;
  const ChainNode* out_chain = chain_sorted_out_edges.first;
  const auto& sorted_out_edges = chain_sorted_out_edges.second;

  ParallelPolicy in_policy = in_chain->parallel_desc()->policy();
  ParallelPolicy out_policy = out_chain->parallel_desc()->policy();
  OpPair (*FwBuildBoxingOp)(size_t in_num, size_t out_num);
  if (in_policy == kDataParallel && out_policy == kDataParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpDataData;
  } else if (in_policy == kDataParallel && out_policy == kModelParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpDataModel;
  } else if (in_policy == kModelParallel && out_policy == kDataParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpModelData;
  } else if (in_policy == kModelParallel && out_policy == kModelParallel) {
    FwBuildBoxingOp = &FwBuildBoxingOpModelModel;
  }

  std::vector<std::string> lbns = FindLbnsBetween(in_chain, out_chain);
  for (const std::string& lbn : lbns) {
    OpPair op_pair = FwBuildBoxingOp(sorted_in_edges.size(), sorted_out_edges.size());
    // First Node
    ExecNode* first_node = mut_exec_gph().NewFinalNode();
    first_node->mut_op() = op_pair.first;
    for (const TaskEdge* edge : sorted_in_edges) {
      first_node->AddConsumedLbnRegstPair(lbn, GetRelatedRegst(edge));
    }
    // Second Node
    ExecNode* second_node = mut_exec_gph().NewFinalNode();
    second_node->mut_op() = op_pair.second;
    for (const TaskEdge* edge : sorted_out_edges) {
      second_node->AddProducedLbnRegstPair(lbn, GetRelatedRegst(edge));
    }
    // Connect
    Connect(first_node, mut_exec_gph().NewExecEdge(lbn), second_node);
  }
}

void BoxingTaskNode::SetProducedRegst() {
  AddInPathLbn2ProducedRegst();
  std::unique_ptr<RegstDesc> boxing_middle_regst(new DisContigRegstDesc);
  for (const std::unique_ptr<ExecEdge>& edge : exec_gph().edges()) {
    boxing_middle_regst->EnrollWithPbnAndLbn(edge->pbn(), edge->lbn());
  }
  AddProducedRegstDesc("boxing_middle", std::move(boxing_middle_regst));
}

namespace {

inline RegstDesc* GetBpRegstFromFwRegst(RegstDesc* fw_regst) {
  const TaskEdge* fw_edge = GetRelatedTaskEdge(fw_regst);
  const TaskEdge* bp_edge = fw_edge->related_fwbp_edge();
  return GetRelatedRegst(bp_edge);
}

}

void BoxingTaskNode::BpBuildExecAndProducedRegsts(Path* path) {
  BindOutEdgeAndRegst();
  const ExecGraph& fw_exec_gph = GetFwNode()->exec_gph();
  HashMap<const ExecNode*, ExecNode*> fw_node2bp_node;
  for (const std::unique_ptr<ExecNode>& fw_node: fw_exec_gph.nodes()) {
    ExecNode* bp_node = mut_exec_gph().NewFinalNode();
    CHECK(fw_node2bp_node.emplace(fw_node.get(), bp_node).second);
    bp_node->mut_op() = fw_node->op();
    for (const auto& fw_pair : fw_node->consumed_lbn_regst_pairs()) {
      RegstDesc* bp_regst = GetBpRegstFromFwRegst(fw_pair.second);
      bp_node->AddProducedLbnRegstPair(fw_pair.first, bp_regst);
    }
    for (const auto& fw_pair : fw_node->produced_lbn_regst_pairs()) {
      RegstDesc* bp_regst = GetBpRegstFromFwRegst(fw_pair.second);
      bp_node->AddConsumedLbnRegstPair(fw_pair.first, bp_regst);
    }
  }
  for (const std::unique_ptr<ExecEdge>& fw_edge : fw_exec_gph.edges()) {
    Connect(fw_node2bp_node.at(fw_edge->dst_node()),
            mut_exec_gph().NewExecEdge(fw_edge->lbn()),
            fw_node2bp_node.at(fw_edge->src_node()));
  }
  mut_exec_gph().UpdateSourceAndSink();
  SetProducedRegst();
}

} // namespace oneflow
