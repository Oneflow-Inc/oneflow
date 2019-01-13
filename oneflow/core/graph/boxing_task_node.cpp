#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

void BoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* out_edge : out_edges()) {
    std::string name = "boxing_out_" + std::to_string(out_edge->edge_id());
    auto out_regst = ProduceRegst(name, true);
    out_edge->AddRegst(name, out_regst);
  }
  ProduceRegst("middle", true, 1, 1);
}

void BoxingTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* in_edge : in_edges()) {
    std::string name = "boxing_in_" + std::to_string(in_edge->edge_id());
    auto in_regst = in_edge->GetSoleRegst();
    ConsumeRegst(name, in_regst);
  }
}

void BoxingTaskNode::BuildExecGphAndRegst() {
  HashMap<const LogicalNode*, std::vector<EdgeInfo>> in_logical2edge_info;
  InitLogical2SortedEdgeInfo(&TaskNode::in_edges, &TaskNode::SoleInEdge, &TaskEdge::src_node,
                             &in_logical2edge_info);
  HashMap<const LogicalNode*, std::vector<EdgeInfo>> out_logical2edge_info;
  InitLogical2SortedEdgeInfo(&TaskNode::out_edges, &TaskNode::SoleOutEdge, &TaskEdge::dst_node,
                             &out_logical2edge_info);
  for (const auto& in_pair : in_logical2edge_info) {
    for (const auto& out_pair : out_logical2edge_info) {
      BuildWithLogicalPair(in_pair.first, in_pair.second, out_pair.first, out_pair.second);
    }
  }
}

#define DEFINE_BLD_BOXING_OP_CONF_METHOD(x, y)                                      \
  void x::BldBoxingOpConfWith##y(                                                   \
      const LogicalBlobId& lbi, const std::vector<EdgeInfo>& sorted_in_edges,       \
      const LogicalNode* in_logical, const std::vector<EdgeInfo>& sorted_out_edges, \
      const LogicalNode* out_logical, BoxingOpConf* conf)

static void SetBoxSplitPart(const std::vector<BoxingTaskNode::EdgeInfo>& sorted_edges,
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
  CHECK_EQ(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi)
               % out_logical->parallel_desc()->parallel_num(),
           0);
  BalancedSplitter bs(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi),
                      out_logical->parallel_desc()->parallel_num());
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(OutBoxingTaskNode, DataConcatAndDataSplit) {
  conf->mutable_concat_box()->set_axis(0);
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(0);
  CHECK_EQ(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi)
               % in_logical->parallel_desc()->parallel_num(),
           0);
  BalancedSplitter in_bs(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi),
                         in_logical->parallel_desc()->parallel_num());
  Range in_range =
      in_bs.At(sorted_in_edges.front().parallel_id_min, sorted_in_edges.back().parallel_id_max);
  CHECK_EQ(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi)
               % out_logical->parallel_desc()->parallel_num(),
           0);
  BalancedSplitter out_bs(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi),
                          out_logical->parallel_desc()->parallel_num());
  for (const EdgeInfo& out_edge : sorted_out_edges) {
    Range out_range = out_bs.At(out_edge.parallel_id_min, out_edge.parallel_id_max);
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
  split_conf->set_axis(out_logical->GetModelSplitAxis());
  BalancedSplitter bs(out_logical->GetMaxModelSplitNum(),
                      out_logical->parallel_desc()->parallel_num());
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, ModelConcatAndDataSplit) {
  conf->mutable_concat_box()->set_axis(in_logical->GetModelSplitAxis());
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(0);
  CHECK_EQ(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi)
               % out_logical->parallel_desc()->parallel_num(),
           0);
  BalancedSplitter bs(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi),
                      out_logical->parallel_desc()->parallel_num());
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, ModelConcatAndClone) {
  conf->mutable_concat_box()->set_axis(in_logical->GetModelSplitAxis());
  conf->mutable_clone_box();
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, AddAndDataSplit) {
  conf->mutable_add_box();
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(0);
  CHECK_EQ(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi)
               % out_logical->parallel_desc()->parallel_num(),
           0);
  BalancedSplitter bs(Global<JobDesc>::Get()->LogicalBlobDim04Lbi(lbi),
                      out_logical->parallel_desc()->parallel_num());
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, AddAndModelSplit) {
  conf->mutable_add_box();
  BoxSplitConf* split_conf = conf->mutable_split_box();
  split_conf->set_axis(out_logical->GetModelSplitAxis());
  BalancedSplitter bs(out_logical->GetMaxModelSplitNum(),
                      out_logical->parallel_desc()->parallel_num());
  SetBoxSplitPart(sorted_out_edges, bs, split_conf);
}
DEFINE_BLD_BOXING_OP_CONF_METHOD(BoxingTaskNode, AddAndClone) {
  conf->mutable_add_box();
  conf->mutable_clone_box();
}

void BoxingTaskNode::InitLogical2SortedEdgeInfo(
    const std::unordered_set<TaskEdge*>& (TaskNode::*GetEdges)() const,
    TaskEdge* (TaskNode::*SoleEdge)() const, TaskNode* (TaskEdge::*SoleNode)() const,
    HashMap<const LogicalNode*, std::vector<EdgeInfo>>* logical2sorted_edge_info) {
  logical2sorted_edge_info->clear();
  for (const TaskEdge* edge : (this->*GetEdges)()) {
    EdgeInfo edge_info;
    edge_info.edge = edge;
    edge_info.parallel_id_min = GetMaxVal<int64_t>();
    edge_info.parallel_id_max = GetMinVal<int64_t>();
    std::queue<const TaskNode*> node_queue;
    node_queue.push((edge->*SoleNode)());
    const LogicalNode* logical = nullptr;
    while (node_queue.empty() == false) {
      const TaskNode* cur_node = node_queue.front();
      node_queue.pop();
      auto cur_comp_node = dynamic_cast<const CompTaskNode*>(cur_node);
      if (cur_comp_node) {
        edge_info.parallel_id_min =
            std::min(edge_info.parallel_id_min, cur_comp_node->parallel_id());
        edge_info.parallel_id_max =
            std::max(edge_info.parallel_id_max, cur_comp_node->parallel_id());
        if (logical == nullptr) { logical = cur_comp_node->logical_node(); }
      } else {
        for (const TaskEdge* cur_edge : (cur_node->*GetEdges)()) {
          node_queue.push((cur_edge->*SoleNode)());
        }
      }
    }
    (*logical2sorted_edge_info)[logical].push_back(edge_info);
  }
  for (auto& pair : *logical2sorted_edge_info) {
    std::vector<EdgeInfo>& edges = pair.second;
    std::sort(edges.begin(), edges.end(), [&](const EdgeInfo& lhs, const EdgeInfo& rhs) {
      return lhs.parallel_id_min < rhs.parallel_id_min;
    });
  }
}

void BoxingTaskNode::BuildWithLogicalPair(const LogicalNode* in_logical,
                                          const std::vector<EdgeInfo>& sorted_in_edges,
                                          const LogicalNode* out_logical,
                                          const std::vector<EdgeInfo>& sorted_out_edges) {
  std::vector<LogicalBlobId> lbis = in_logical->GetLbisTo(out_logical);
  auto middle_regst = GetProducedRegst("middle");
  for (const LogicalBlobId& lbi : lbis) {
    ExecNode* node = mut_exec_gph().NewNode();
    node->mut_op() = NewBoxingOp(lbi, in_logical, out_logical, sorted_in_edges, sorted_out_edges);
    for (size_t i = 0; i < node->op()->input_bns().size(); ++i) {
      auto regst = sorted_in_edges[i].edge->GetSoleRegst();
      const std::string& ibn = node->op()->input_bns().Get(i);
      node->BindBnWithRegst(ibn, regst);
    }
    for (size_t i = 0; i < node->op()->output_bns().size(); ++i) {
      auto regst = sorted_out_edges[i].edge->GetSoleRegst();
      const std::string& obn = node->op()->output_bns().Get(i);
      if (lbi.is_packed_id()) {
        RegstDesc* in_regst = sorted_in_edges[0].edge->GetSoleRegst().get();
        if (!regst->HasSameBlobDescs(in_regst)) { regst->CopyBlobDescFrom(in_regst); }
      } else {
        regst->AddLbi(lbi);
      }
      node->BindBnWithRegst(obn, regst);
    }
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      CHECK_EQ(lbi.is_packed_id(), false);
      middle_regst->AddLbi(node->op()->BnInOp2Lbi(dtbn));
      node->BindBnWithRegst(dtbn, middle_regst);
    }
    if (lbi.is_packed_id() == false) { node->InferBlobDescs(nullptr); }
  }
}

std::shared_ptr<Operator> BoxingTaskNode::NewBoxingOp(
    const LogicalBlobId& lbi, const LogicalNode* in_logical, const LogicalNode* out_logical,
    const std::vector<EdgeInfo>& sorted_in_edges, const std::vector<EdgeInfo>& sorted_out_edges) {
  BldBoxingOpConfMthd method = GetMthdForBldBoxingOpConf(in_logical, out_logical);
  OperatorConf op_conf;
  op_conf.set_name("boxing_op_" + NewUniqueId());
  op_conf.set_device_type(device_type());
  BoxingOpConf* boxing_conf = op_conf.mutable_boxing_conf();
  *(boxing_conf->mutable_lbi()) = lbi;
  boxing_conf->set_in_num(sorted_in_edges.size());
  boxing_conf->set_out_num(sorted_out_edges.size());
  (this->*method)(lbi, sorted_in_edges, in_logical, sorted_out_edges, out_logical, boxing_conf);
  return ConstructOp(op_conf);
}

void BoxingTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

}  // namespace oneflow
