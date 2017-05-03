#include "graph/boxing_task_node.h"
#include <algorithm>
#include "operator/operator_manager.h"
#include "operator/boxing_op.h"
#include "graph/comp_task_node.h"

namespace oneflow {

namespace {

void FwCompleteBoxOpConfDataData(BoxingOpConf* conf) {
  conf->mutable_concat_box()->set_axis(gDataConcatInitAxis);
  conf->mutable_data_split_box();
}

void FwCompleteBoxOpConfDataModel(BoxingOpConf* conf) {
  conf->mutable_concat_box()->set_axis(gDataConcatInitAxis);
  conf->mutable_clone_box();
}

void FwCompleteBoxOpConfModelData(BoxingOpConf* conf) {
  conf->mutable_concat_box()->set_axis(gModelConcatInitAxis);
  conf->mutable_data_split_box();
}

void FwCompleteBoxOpConfModelModel(BoxingOpConf* conf) {
  conf->mutable_concat_box()->set_axis(gModelConcatInitAxis);
  conf->mutable_clone_box();
}

} // namespace

void BoxingTaskNode::FwBuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  EnrollAllRegstAndBindRelatedEdge();
  FwVirtualBuild();
}

void BoxingTaskNode::EnrollAllRegstAndBindRelatedEdge() {
  for (TaskEdge* edge : out_edges()) {
    std::string name = "boxing_out_" + edge->edge_id_str();
    auto regst_desc = RegstDescMgr::Singleton().CreateRegisterDesc();
    BindProducedRegstAndOutEdge(regst_desc.get(), edge);
    EnrollProducedRegstDesc(name, std::move(regst_desc));
  }
  auto regst_desc = RegstDescMgr::Singleton().CreateRegisterDesc();
  EnrollProducedRegstDesc("middle", std::move(regst_desc));
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
  void (*CompleteBoxOp)(BoxingOpConf*);
  if (in_policy == kDataParallel && out_policy == kDataParallel) {
    CompleteBoxOp = &FwCompleteBoxOpConfDataData;
  } else if (in_policy == kDataParallel && out_policy == kModelParallel) {
    CompleteBoxOp = &FwCompleteBoxOpConfDataModel;
  } else if (in_policy == kModelParallel && out_policy == kDataParallel) {
    CompleteBoxOp = &FwCompleteBoxOpConfModelData;
  } else {
    CompleteBoxOp = &FwCompleteBoxOpConfModelModel;
  }

  RegstDesc* middle_regst = GetProducedRegstDesc("middle");
  std::vector<std::string> lbns = FindLbnsBetween(in_chain, out_chain);
  for (const std::string& lbn : lbns) {
    // Construct Op
    OperatorConf op_conf;
    op_conf.set_name("boxing_op_" + NewUniqueId());
    BoxingOpConf* box_conf = op_conf.mutable_boxing_conf();
    box_conf->set_lbn(lbn);
    box_conf->set_in_num(sorted_in_edges.size());
    box_conf->set_out_num(sorted_out_edges.size());
    CompleteBoxOp(box_conf);
    ExecNode* node = mut_exec_gph().NewNode();
    node->mut_op() = OpMgr::Singleton().ConstructOp(op_conf);
    // ibn
    for (size_t i = 0; i < sorted_in_edges.size(); ++i) {
      RegstDesc* in_regst = GetRelatedRegst(sorted_in_edges.at(i));
      const std::string& ibn = node->op()->input_bns().at(i);
      node->BindBnInOpAndRegst(ibn, in_regst);
    }
    // obn
    for (size_t i = 0; i < sorted_out_edges.size(); ++i) {
      RegstDesc* out_regst = GetRelatedRegst(sorted_out_edges.at(i));
      const std::string& obn = node->op()->output_bns().at(i);
      std::string lbn = node->op()->Lbn4BnInOp(obn);
      out_regst->EnrollLbn(lbn);
      node->BindBnInOpAndRegst(obn, out_regst);
    }
    // dtbn
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      std::string lbn = node->op()->Lbn4BnInOp(dtbn);
      middle_regst->EnrollLbn(lbn);
      node->BindBnInOpAndRegst(dtbn, middle_regst);
    }
  }
}

void BoxingTaskNode::FwInferShape4LbnInProducedRegsts(TaskGraph*) {
  for (const auto& exec_node : exec_gph().nodes()) {
    exec_node->op()->InferShape4FwBlobs(
        exec_node->GetMutShapePtr4BnInOpFunc(),
        chain_node()->parallel_desc()->policy(),
        0,
        0);
  }
}

namespace {

inline RegstDesc* GetBpRegstFromFwRegst(RegstDesc* fw_regst) {
  const TaskEdge* fw_edge = GetRelatedTaskEdge(fw_regst);
  const TaskEdge* bp_edge = fw_edge->related_fwbp_edge();
  return GetRelatedRegst(bp_edge);
}

}

void BoxingTaskNode::BpBuildExecAndEnrollLbn2Regsts(TaskGraph*) {
  EnrollAllRegstAndBindRelatedEdge();
  const ExecGraph& fw_exec_gph = GetFwNode()->exec_gph();
  for (const std::unique_ptr<ExecNode>& fw_node: fw_exec_gph.nodes()) {
    ExecNode* bp_node = mut_exec_gph().NewNode();
    bp_node->mut_op() = fw_node->op();
    // in_diff
    for (const std::string& ibn : fw_node->op()->input_bns()) {
      std::string idbn = GenDiffBn(ibn);
      std::string lbn = fw_node->op()->Lbn4BnInOp(ibn);
      RegstDesc* in_regst = fw_node->GetRegstFromBnInOp(ibn);
      RegstDesc* in_diff_regst = GetBpRegstFromFwRegst(in_regst);
      in_diff_regst->EnrollLbn(lbn);
      bp_node->BindBnInOpAndRegst(idbn, in_diff_regst);
    }
    // out_diff
    for (const std::string& obn : fw_node->op()->output_bns()) {
      std::string odbn = GenDiffBn(obn);
      std::string lbn = fw_node->op()->Lbn4BnInOp(obn);
      RegstDesc* out_regst = fw_node->GetRegstFromBnInOp(obn);
      RegstDesc* out_diff_regst = GetBpRegstFromFwRegst(out_regst);
      bp_node->BindBnInOpAndRegst(odbn, out_diff_regst);
    }
    // data tmp
    for (const std::string& dtbn : fw_node->op()->data_tmp_bns()) {
      std::string lbn = fw_node->op()->Lbn4BnInOp(dtbn);
      RegstDesc* bp_middle_regst = GetProducedRegstDesc("middle");
      bp_middle_regst->EnrollLbn(lbn);
      bp_node->BindBnInOpAndRegst(dtbn, bp_middle_regst);
    }
  }
  mut_exec_gph().UpdateSourceAndSink();
}
  
void BoxingTaskNode::BpInferShape4LbnInProducedRegsts(TaskGraph*) {
  for (TaskEdge* fw_in_edge : GetFwNode()->in_edges()) {
    RegstDesc* in_regst = GetRelatedRegst(fw_in_edge);
    RegstDesc* in_diff_regst = GetBpRegstFromFwRegst(in_regst);
    in_diff_regst->CopyShapeFrom(in_regst);
  }
  RegstDesc* fw_middle_regst = GetFwNode()->GetProducedRegstDesc("middle");
  RegstDesc* bp_middle_regst = GetProducedRegstDesc("middle");
  bp_middle_regst->CopyShapeFrom(fw_middle_regst);
}

} // namespace oneflow
