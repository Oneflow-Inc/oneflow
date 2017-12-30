#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

void BoxingTaskNode::Init(int64_t machine_id) {
  set_machine_id(machine_id);
  set_thrd_id(IDMgr::Singleton()->AllocateBoxingThrdId(machine_id));
}

void BoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* out_edge : out_edges()) {
    std::string name = "boxing_out_" + std::to_string(out_edge->edge_id());
    auto out_regst = ProduceRegst(name);
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

void BoxingTaskNode::BuildExecGphAndRegst() {
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

#define DEFINE_BLD_BOXING_OP_CONF_METHOD(x, y)                                \
  void x::BldBoxingOpConfWith##y(                                             \
      const std::string& lbn, const std::vector<EdgeInfo>& sorted_in_edges,   \
      int64_t in_parallel_num, const std::vector<EdgeInfo>& sorted_out_edges, \
      int64_t out_parallel_num, BoxingOpConf* conf)

static void SetBoxSplitPart(
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_edges,
    const BalancedSplitter& bs, BoxSplitConf* split_conf) {
  for (const BoxingTaskNode::EdgeInfo& edge_info : sorted_edges) {
    Range range = bs.At(edge_info.parallel_id_min, edge_info.parallel_id_max);
    split_conf->add_part_num(range.size());
  }
}

DEFINE_BLD_BOXING_OP_CONF_METHOD(InBoxingTaskNode, DataConcatAndDataSplit) {
  conf->mutable_concat_box()->set_axis(0);
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(0);
  BalancedSplitter bs(JobDesc::Singleton()->ParallelPieceSize(),
                      out_parallel_num);
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(OutBoxingTaskNode, DataConcatAndDataSplit) {
  conf->mutable_concat_box()->set_axis(0);
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(0);
  BalancedSplitter in_bs(JobDesc::Singleton()->ParallelPieceSize(),
                         in_parallel_num);
  Range in_range = in_bs.At(sorted_in_edges.front().parallel_id_min,
                            sorted_in_edges.back().parallel_id_max);
  BalancedSplitter out_bs(JobDesc::Singleton()->ParallelPieceSize(),
                          out_parallel_num);
  for (const EdgeInfo& out_edge : sorted_out_edges) {
    Range out_range =
        out_bs.At(out_edge.parallel_id_min, out_edge.parallel_id_max);
    Range intersectant_range = FindIntersectant(in_range, out_range);
    split_conf->add_part_num(intersectant_range.size());
  }
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, DataConcatAndClone) {
  conf->mutable_concat_box()->set_axis(0);
  conf->mutable_clone_box();
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, DataConcatAndModelSplit) {
  conf->mutable_concat_box()->set_axis(0);
  BoxSplitConf* split_conf = conf->mutable_split_box();
  auto producer_op = LogicalGraph::Singleton()->GetProducerOp(lbn);
  split_conf->set_axis(producer_op->ModelSplitAxis());
  BalancedSplitter bs(producer_op->MaxModelSplitNum(), out_parallel_num);
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, ModelConcatAndDataSplit) {
  auto producer_op = LogicalGraph::Singleton()->GetProducerOp(lbn);
  conf->mutable_concat_box()->set_axis(producer_op->ModelSplitAxis());
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(0);
  BalancedSplitter bs(JobDesc::Singleton()->ParallelPieceSize(),
                      out_parallel_num);
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, ModelConcatAndClone) {
  auto producer_op = LogicalGraph::Singleton()->GetProducerOp(lbn);
  conf->mutable_concat_box()->set_axis(producer_op->ModelSplitAxis());
  conf->mutable_clone_box();
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, AddAndDataSplit) {
  conf->mutable_add_box();
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(0);
  BalancedSplitter bs(JobDesc::Singleton()->ParallelPieceSize(),
                      out_parallel_num);
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, AddAndModelSplit) {
  auto producer_op = LogicalGraph::Singleton()->GetProducerOp(lbn);
  conf->mutable_add_box();
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(producer_op->ModelSplitAxis());
  BalancedSplitter bs(producer_op->MaxModelSplitNum(), out_parallel_num);
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, AddAndClone) {
  conf->mutable_add_box();
  conf->mutable_clone_box();
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
    node->mut_op() = NewBoxingOp(lbn, in_chain, out_chain, sorted_in_edges,
                                 sorted_out_edges);
    for (size_t i = 0; i < node->op()->input_bns().size(); ++i) {
      auto regst = sorted_in_edges[i].edge->GetSoleRegst();
      const std::string& ibn = node->op()->input_bns().at(i);
      node->BindBnInOpAndRegst(ibn, regst);
    }
    for (size_t i = 0; i < node->op()->output_bns().size(); ++i) {
      auto regst = sorted_out_edges[i].edge->GetSoleRegst();
      const std::string& obn = node->op()->output_bns().at(i);
      if (lbn == kPackedBlobName) {
        regst->CopyBlobDescFrom(sorted_in_edges[0].edge->GetSoleRegst().get());
      } else {
        regst->AddLbn(lbn);
      }
      node->BindBnInOpAndRegst(obn, regst);
    }
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      CHECK_STRNE(lbn.c_str(), kPackedBlobName);
      middle_regst->AddLbn(node->op()->Lbn4BnInOp(dtbn));
      node->BindBnInOpAndRegst(dtbn, middle_regst);
    }
    if (lbn != kPackedBlobName) {
      node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), nullptr);
    }
  }
}

std::shared_ptr<Operator> BoxingTaskNode::NewBoxingOp(
    const std::string& lbn, const ChainNode* in_chain,
    const ChainNode* out_chain, const std::vector<EdgeInfo>& sorted_in_edges,
    const std::vector<EdgeInfo>& sorted_out_edges) {
  BldBoxingOpConfMthd method = in_chain->GetMthdForBldBoxingOpConfTo(out_chain);
  OperatorConf op_conf;
  op_conf.set_name("boxing_op_" + NewUniqueId());
  BoxingOpConf* boxing_conf = op_conf.mutable_boxing_conf();
  boxing_conf->set_lbn(lbn);
  boxing_conf->set_in_num(sorted_in_edges.size());
  boxing_conf->set_out_num(sorted_out_edges.size());
  (this->*method)(lbn, sorted_in_edges,
                  in_chain->parallel_desc()->parallel_num(), sorted_out_edges,
                  out_chain->parallel_desc()->parallel_num(), boxing_conf);
  return ConstructOp(op_conf);
}

}  // namespace oneflow
