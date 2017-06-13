#include "oneflow/core/graph/boxing_task_node.h"
#include <algorithm>
#include "oneflow/core/operator/operator_manager.h"
#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

namespace {

void FwCompleteBoxOpConfDataData(BoxingOpConf* conf) {
  conf->mutable_concat_box()->set_axis(0);
  conf->mutable_data_split_box();
}

void FwCompleteBoxOpConfDataModel(BoxingOpConf* conf) {
  conf->mutable_concat_box()->set_axis(0);
  conf->mutable_clone_box();
}

void FwCompleteBoxOpConfModelData(BoxingOpConf* conf) {
  conf->mutable_concat_box()->set_axis(1);
  conf->mutable_data_split_box();
}

void FwCompleteBoxOpConfModelModel(BoxingOpConf* conf) {
  conf->mutable_concat_box()->set_axis(1);
  conf->mutable_clone_box();
}

void FwCompleteBoxOpConfFakerMdUpdt(BoxingOpConf* conf) {
  conf->mutable_add_box();
  conf->mutable_clone_box();
}

} // namespace

void BoxingTaskNode::FwBuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  EnrollAllRegstAndBindRelatedEdge();
  FwVirtualBuild();
}

void BoxingTaskNode::EnrollAllRegstAndBindRelatedEdge() {
  for (TaskEdge* edge : in_edges()) {
    std::string name = "boxing_in_" + edge->edge_id_str();
    SubscribeRegstDesc(name, GetRelatedRegst(edge));
  }
  for (TaskEdge* edge : out_edges()) {
    std::string name = "boxing_out_" + edge->edge_id_str();
    auto regst_desc = NewProducedRegstDesc(name);
    BindProducedRegstAndOutEdge(regst_desc, edge);
  }
  NewProducedRegstDesc("middle");
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
  // useful vars
  const ChainNode* in_chain = chain_sorted_in_edges.first;
  const auto& sorted_in_edges = chain_sorted_in_edges.second;
  const ChainNode* out_chain = chain_sorted_out_edges.first;
  const auto& sorted_out_edges = chain_sorted_out_edges.second;
  // 4 case
  ParallelPolicy in_policy = in_chain->parallel_desc()->policy();
  ParallelPolicy out_policy = out_chain->parallel_desc()->policy();
  void (*CompleteBoxOp)(BoxingOpConf*);
  if (in_policy == kDataParallel && out_policy == kDataParallel) {
    CompleteBoxOp = &FwCompleteBoxOpConfDataData;
  } else if (in_policy == kDataParallel && out_policy == kModelParallel) {
    CompleteBoxOp = &FwCompleteBoxOpConfDataModel;
  } else if (in_policy == kModelParallel && out_policy == kDataParallel) {
    CompleteBoxOp = &FwCompleteBoxOpConfModelData;
  } else if (in_policy == kModelParallel && out_policy == kModelParallel) {
    CompleteBoxOp = &FwCompleteBoxOpConfModelModel;
  } else {
    CHECK_EQ(in_policy, kFakerMdUpdt);
    CHECK_EQ(out_policy, kModelParallel);
    CompleteBoxOp = &FwCompleteBoxOpConfFakerMdUpdt;
  }
  // func 4 construct boxing_op in this node
  auto ConstructBoxingOp = [&](const std::string& lbn) {
    OperatorConf op_conf;
    op_conf.set_name("boxing_op_" + NewUniqueId());
    BoxingOpConf* box_conf = op_conf.mutable_boxing_conf();
    box_conf->set_lbn(lbn);
    box_conf->set_in_num(sorted_in_edges.size());
    box_conf->set_out_num(sorted_out_edges.size());
    CompleteBoxOp(box_conf);
    return OpMgr::Singleton().ConstructOp(op_conf);
  };
  // lbns
  std::vector<std::string> lbns = FindLbnsBetween(in_chain, out_chain);
  if (lbns.at(0) == kBaledBlobName) {
    CHECK_EQ(lbns.size(), 1);
    lbns.clear();
    auto in_regst_0 = GetRelatedRegst(sorted_in_edges.at(0));
    in_regst_0->ForEachLbn([&](const std::string& lbn) {
      lbns.push_back(lbn);
    });
  }
  // Enroll Lbn
  auto middle_regst = GetProducedRegstDesc("middle");
  for (const std::string& lbn : lbns) {
    ExecNode* node = mut_exec_gph().NewNode();
    node->mut_op() = ConstructBoxingOp(lbn);
    // ibn
    for (size_t i = 0; i < sorted_in_edges.size(); ++i) {
      auto in_regst = GetRelatedRegst(sorted_in_edges.at(i));
      const std::string& ibn = node->op()->input_bns().at(i);
      node->BindBnInOpAndRegst(ibn, in_regst);
    }
    // obn
    for (size_t i = 0; i < sorted_out_edges.size(); ++i) {
      auto out_regst = GetRelatedRegst(sorted_out_edges.at(i));
      const std::string& obn = node->op()->output_bns().at(i);
      out_regst->EnrollLbn(lbn);
      node->BindBnInOpAndRegst(obn, out_regst);
    }
    // dtbn
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      middle_regst->EnrollLbn(node->op()->Lbn4BnInOp(dtbn));
      node->BindBnInOpAndRegst(dtbn, middle_regst);
    }
  }
}

void BoxingTaskNode::FwInferShapeOfBlobsInProducedRegsts(TaskGraph*) {
  exec_gph().ConstForEachNode([this](const ExecNode* exec_node) {
    exec_node->op()->InferShape4FwBlobs(
        exec_node->GetMutShapePtr4BnInOpFunc(),
        chain_node()->parallel_desc()->policy(),
        0,
        0);
  });
}

namespace {

std::shared_ptr<RegstDesc> GetBpRegstFromFwRegst(
    std::shared_ptr<RegstDesc> fw_regst) {
  const TaskEdge* fw_edge = GetRelatedTaskEdge(fw_regst);
  const TaskEdge* bp_edge = fw_edge->related_fwbp_edge();
  if (bp_edge == nullptr) { return nullptr; }
  return GetRelatedRegst(bp_edge);
}

}

void BoxingTaskNode::BpBuildExecAndEnrollLbn2Regsts(TaskGraph*) {
  EnrollAllRegstAndBindRelatedEdge();
  GetFwNode()->exec_gph().ConstForEachNode([&](const ExecNode* fw_node) {
    std::unique_ptr<ExecNode> bp_node(new ExecNode);
    bp_node->mut_op() = fw_node->op();
    bool need_enroll = true;
    // in_diff
    for (const std::string& ibn : fw_node->op()->input_bns()) {
      std::string idbn = GenDiffBn(ibn);
      std::string lbn = fw_node->op()->Lbn4BnInOp(ibn);
      auto in_regst = fw_node->GetRegstFromBnInOp(ibn);
      auto in_diff_regst = GetBpRegstFromFwRegst(in_regst);
      if (!in_diff_regst) {
        need_enroll = false;
        break;
      }
      in_diff_regst->EnrollLbn(lbn);
      bp_node->BindBnInOpAndRegst(idbn, in_diff_regst);
    }
    if (need_enroll == false) { return; }
    // out_diff
    for (const std::string& obn : fw_node->op()->output_bns()) {
      std::string odbn = GenDiffBn(obn);
      std::string lbn = fw_node->op()->Lbn4BnInOp(obn);
      auto out_regst = fw_node->GetRegstFromBnInOp(obn);
      auto out_diff_regst = GetBpRegstFromFwRegst(out_regst);
      bp_node->BindBnInOpAndRegst(odbn, out_diff_regst);
    }
    // data tmp
    for (const std::string& dtbn : fw_node->op()->data_tmp_bns()) {
      std::string lbn = fw_node->op()->Lbn4BnInOp(dtbn);
      auto bp_middle_regst = GetProducedRegstDesc("middle");
      bp_middle_regst->EnrollLbn(lbn);
      bp_node->BindBnInOpAndRegst(dtbn, bp_middle_regst);
    }
    mut_exec_gph().EnrollNode(std::move(bp_node));
  });
  mut_exec_gph().UpdateSourceAndSink();
}
  
void BoxingTaskNode::BpInferShapeOfBlobsInProducedRegsts(TaskGraph*) {
  for (TaskEdge* fw_in_edge : GetFwNode()->in_edges()) {
    auto in_regst = GetRelatedRegst(fw_in_edge);
    if (auto in_diff_regst = GetBpRegstFromFwRegst(in_regst)) {
      in_diff_regst->CopyShapeFrom(in_regst.get());
    }
  }
  auto fw_middle_regst = GetFwNode()->GetProducedRegstDesc("middle");
    auto bp_middle_regst = GetProducedRegstDesc("middle");
  bp_middle_regst->CopyShapeFrom(fw_middle_regst.get());
}

} // namespace oneflow
