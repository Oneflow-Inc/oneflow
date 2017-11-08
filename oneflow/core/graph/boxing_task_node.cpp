#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void BoxingTaskNode::Init(int64_t machine_id) {
  set_machine_id(machine_id);
  set_thrd_loc_id(IDMgr::Singleton()->BoxingThrdLocId());
}

void BoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* out_edge : out_edges()) {
    std::string name = "boxing_out_" + std::to_string(out_edge->edge_id());
    auto out_regst = ProduceRegst(name, 1, kMaxRegisterNum);
    out_edge->AddRegst(name, out_regst);
  }
  ProduceRegst("middle", 1, 1);
}

void BoxingTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* in_edge : in_edges()) {
    std::string name = "boxing_in_" + std::to_string(in_edge->edge_id());
    auto in_regst = in_edge->GetSoleRegst();
    ConsumeRegst(name, in_regst);
  }
}

void BoxingTaskNode::Build() {
  HashMap<const ChainNode*, std::vector<EdgeInfo>> in_chain2edge_info;
  InitChain2SortedEdgeInfo(&TaskNode::in_edges, &TaskNode::SoleInEdge,
                           &TaskEdge::src_node, &in_chain2edge_info);
  HashMap<const ChainNode*, std::vector<EdgeInfo>> out_chain2edge_info;
  InitChain2SortedEdgeInfo(&TaskNode::out_edges, &TaskNode::SoleOutEdge,
                           &TaskEdge::dst_node, &out_chain2edge_info);
  for (const auto& in_pair : in_chain2edge_info) {
    for (const auto& out_pair : out_chain2edge_info) {
      BuildWithChainPair(in_pair.first, in_pair.second, out_pair.first,
                         out_pair.second);
    }
  }
}

std::shared_ptr<Operator> BoxingTaskNode::BldBoxingOpWithDataConcatAndDataSplit(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t* used_in_edge_begin, int64_t* used_out_edge_begin) {
  TODO();
}
std::shared_ptr<Operator> BoxingTaskNode::BldBoxingOpWithDataConcatAndClone(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t* used_in_edge_begin, int64_t* used_out_edge_begin) {
  TODO();
}
std::shared_ptr<Operator>
BoxingTaskNode::BldBoxingOpWithDataConcatAndModelSplit(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t* used_in_edge_begin, int64_t* used_out_edge_begin) {
  TODO();
}
std::shared_ptr<Operator>
BoxingTaskNode::BldBoxingOpWithModelConcatAndDataSplit(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t* used_in_edge_begin, int64_t* used_out_edge_begin) {
  TODO();
}
std::shared_ptr<Operator> BoxingTaskNode::BldBoxingOpWithModelConcatAndClone(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t* used_in_edge_begin, int64_t* used_out_edge_begin) {
  TODO();
}
std::shared_ptr<Operator> BoxingTaskNode::BldBoxingOpWithAddAndDataSplit(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t* used_in_edge_begin, int64_t* used_out_edge_begin) {
  TODO();
}
std::shared_ptr<Operator> BoxingTaskNode::BldBoxingOpWithAddAndModelSplit(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t* used_in_edge_begin, int64_t* used_out_edge_begin) {
  TODO();
}
std::shared_ptr<Operator> BoxingTaskNode::BldBoxingOpWithAddAndClone(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t* used_in_edge_begin, int64_t* used_out_edge_begin) {
  TODO();
}

void BoxingTaskNode::InitChain2SortedEdgeInfo(
    const std::unordered_set<TaskEdge*>& (TaskNode::*GetEdges)() const,
    TaskEdge* (TaskNode::*SoleEdge)() const,
    TaskNode* (TaskEdge::*SoleNode)() const,
    HashMap<const ChainNode*, std::vector<EdgeInfo>>* chain2sorted_edge_info) {
  chain2sorted_edge_info->clear();
  for (const TaskEdge* edge : (this->*GetEdges)()) {
    EdgeInfo edge_info;
    edge_info.edge = edge;
    edge_info.parallel_id_min = std::numeric_limits<int64_t>::max();
    edge_info.parallel_id_max = std::numeric_limits<int64_t>::min();
    std::queue<const TaskNode*> node_queue;
    node_queue.push((edge->*SoleNode)());
    const ChainNode* chain = nullptr;
    while (node_queue.empty() == false) {
      const TaskNode* cur_node = node_queue.front();
      node_queue.pop();
      auto cur_comp_node = dynamic_cast<const CompTaskNode*>(cur_node);
      if (cur_comp_node) {
        edge_info.parallel_id_min =
            std::min(edge_info.parallel_id_min, cur_comp_node->parallel_id());
        edge_info.parallel_id_max =
            std::max(edge_info.parallel_id_max, cur_comp_node->parallel_id());
        if (chain == nullptr) { chain = cur_comp_node->chain_node(); }
      } else {
        for (const TaskEdge* cur_edge : (cur_node->*GetEdges)()) {
          node_queue.push((cur_edge->*SoleNode)());
        }
      }
    }
    (*chain2sorted_edge_info)[chain].push_back(edge_info);
  }
  for (auto& pair : *chain2sorted_edge_info) {
    std::vector<EdgeInfo>& edges = pair.second;
    std::sort(edges.begin(), edges.end(),
              [&](const EdgeInfo& lhs, const EdgeInfo& rhs) {
                return lhs.parallel_id_min < rhs.parallel_id_min;
              });
  }
}

void BoxingTaskNode::BuildWithChainPair(
    const ChainNode* in_chain, const std::vector<EdgeInfo>& sorted_in_edges,
    const ChainNode* out_chain, const std::vector<EdgeInfo>& sorted_out_edges) {
  std::vector<std::string> lbns = FindLbnsBetween(in_chain, out_chain);
  auto middle_regst = GetProducedRegst("middle");
  for (const std::string& lbn : lbns) {
    ExecNode* node = mut_exec_gph().NewNode();
    int64_t used_in_edge_begin = -1;
    int64_t used_out_edge_begin = -1;
    node->mut_op() =
        NewBoxingOp(in_chain, out_chain, sorted_in_edges, sorted_out_edges,
                    &used_in_edge_begin, &used_out_edge_begin);
    CHECK_NE(used_in_edge_begin, -1);
    CHECK_NE(used_out_edge_begin, -1);
    for (size_t i = 0; i < node->op()->input_bns().size(); ++i) {
      auto regst = sorted_in_edges[used_in_edge_begin + i].edge->GetSoleRegst();
      const std::string& ibn = node->op()->input_bns().at(i);
      node->BindBnInOpAndRegst(ibn, regst);
    }
    for (size_t i = 0; i < node->op()->output_bns().size(); ++i) {
      auto regst =
          sorted_out_edges[used_out_edge_begin + i].edge->GetSoleRegst();
      const std::string& obn = node->op()->output_bns().at(i);
      regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, regst);
    }
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      middle_regst->AddLbn(node->op()->Lbn4BnInOp(dtbn));
      node->BindBnInOpAndRegst(dtbn, middle_regst);
    }
    node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), nullptr);
  }
}

std::shared_ptr<Operator> BoxingTaskNode::NewBoxingOp(
    const ChainNode* in_chain, const ChainNode* out_chain,
    const std::vector<EdgeInfo>& sorted_in_edges,
    const std::vector<EdgeInfo>& sorted_out_edges, int64_t* used_in_edge_begin,
    int64_t* used_out_edge_begin) {
  BldBoxingOpMthd method = in_chain->GetMthdForBldBoxingOpTo(out_chain);
  return (this->*method)(sorted_in_edges, sorted_out_edges, used_in_edge_begin,
                         used_out_edge_begin);
}

}  // namespace oneflow
