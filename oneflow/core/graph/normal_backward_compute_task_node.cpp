#include "oneflow/core/graph/normal_backward_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NormalBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("in_diff", true);
  ProduceRegst("activation_diff", true, 1, 1);
  ProduceRegst("bw_buf", true, 1, 1);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() == "MdDiffAcc" || succ_logical->TypeName() == "NormalMdUpdt"
        || succ_logical->TypeName() == "ReduceScatter" || succ_logical->TypeName() == "ReduceConcat"
        || succ_logical->TypeName() == "NcclAllReduce"
        || succ_logical->TypeName() == "NcclReduceScatter") {
      edge->AddRegst("model_diff", ProduceRegst("model_diff", true));
      type_name4model_related_logical_node_ = succ_logical->TypeName();
    } else {
      BindEdgeWithProducedRegst(edge, "in_diff");
    }
  }
}

void NormalBackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskType src_task_type = src_node->GetTaskType();
    if (IsForwardTaskType(src_task_type)) {
      ConsumeRegst("activation", edge->GetRegst("activation"));
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
      ConsumeRegst("const_buf", edge->GetRegst("const_buf"));
    } else if (src_task_type == TaskType::kNormalMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("const_model", edge->GetRegst("const_model"));
    } else {
      ConsumeRegst("out_diff", edge->GetSoleRegst());
    }
  }
  CompTaskNode* fw_task = GetRelatedFwTaskNode();
  if (fw_task) {
    const std::list<std::shared_ptr<RegstDesc>>& in_regst = fw_task->GetConsumedRegst("in");
    for (std::shared_ptr<RegstDesc> regst : in_regst) { ConsumeRegst("in", regst); }
  }
}

void NormalBackwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphAndBindOutDiffRegst();
  LinkFwExecNode();
  BuildActivationDiffRegst();
  BuildInDiffRegst();
  BindInRegst();
  BindModelDiffRegst();
  InferBlobDescsInProducedRegsts();
}

void NormalBackwardCompTaskNode::BuildExecGphAndBindOutDiffRegst() {
  HashMap<LogicalBlobId, std::pair<ExecNode*, std::string>> lbi2producer;
  for (std::shared_ptr<const Operator> op : logical_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& idbn : op->input_diff_bns()) {
      const LogicalBlobId& lbi = op->BnInOp2Lbi(idbn);
      CHECK(lbi2producer.insert({lbi, {cur_node, idbn}}).second);
    }
  }
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& odbn : cur_node->op()->output_diff_bns()) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(odbn);
      auto producer_it = lbi2producer.find(lbi);
      if (producer_it != lbi2producer.end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbi(lbi);
        edge->mut_src_bn() = producer_it->second.second;
        edge->mut_dst_bn() = odbn;
        Connect(producer_it->second.first, edge, cur_node);
      } else {
        cur_node->BindBnWithOneOfTheRegsts(odbn, GetConsumedRegst("out_diff"));
      }
    }
  });
  CompTaskNode* fw_task = GetRelatedFwTaskNode();
  if (fw_task) {
    std::shared_ptr<RegstDesc> out_regst = GetSoleConsumedRegst("out");
    mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
      for (const std::string& odbn : cur_node->op()->output_diff_bns()) {
        const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(odbn);
        if (lbi2producer.find(lbi) == lbi2producer.end()) {
          std::string obn =
              GenUnDiffBn(odbn);  // the lbis of obn and odbn may be different, use obn directly
          LogicalBlobId obn_lbi = cur_node->op()->BnInOp2Lbi(obn);
          CHECK(fw_task->logical_node()->IsDataLbiOnOutEdge(obn_lbi) == true);
          cur_node->BindBnWithRegst(GenUnDiffBn(odbn), out_regst);
        }
      }
    });
  }
}

void NormalBackwardCompTaskNode::LinkFwExecNode() {
  CompTaskNode* fw_task = GetRelatedFwTaskNode();
  if (fw_task == nullptr) { return; }
  HashMap<std::string, ExecNode*> op_name2fw_exec;
  fw_task->exec_gph().ForEachNode([&](ExecNode* fw_exec) {
    CHECK(op_name2fw_exec.emplace(fw_exec->op()->op_name(), fw_exec).second);
  });
  mut_exec_gph().ForEachNode([&](ExecNode* bw_exec) {
    auto fw_exec_it = op_name2fw_exec.find(bw_exec->op()->op_name());
    if (fw_exec_it != op_name2fw_exec.end()) { bw_exec->set_fw_node(fw_exec_it->second); }
  });
}

void NormalBackwardCompTaskNode::BuildActivationDiffRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetSoleConsumedRegst("activation");
  std::shared_ptr<RegstDesc> activation_diff_regst = GetProducedRegst("activation_diff");
  mut_exec_gph().ForEachEdge([&](ExecEdge* edge) {
    edge->src_node()->BindBnWithRegst(edge->src_bn(), activation_diff_regst);
    edge->dst_node()->BindBnWithRegst(edge->dst_bn(), activation_diff_regst);
    activation_diff_regst->AddLbi(edge->lbi());
    edge->src_node()->BindBnWithRegst(GenUnDiffBn(edge->src_bn()), activation_regst);
    edge->dst_node()->BindBnWithRegst(GenUnDiffBn(edge->dst_bn()), activation_regst);
  });
}

void NormalBackwardCompTaskNode::BuildInDiffRegst() {
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<LogicalBlobId> found_lbis;
    for (ExecEdge* out_edge : cur_node->out_edges()) {
      CHECK(found_lbis.insert(out_edge->lbi()).second);
    }
    for (const std::string& idbn : cur_node->op()->input_diff_bns()) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(idbn);
      if (logical_node()->IsDataLbiOnOutEdge(lbi)) {
        in_diff_regst->AddLbi(lbi);
        cur_node->BindBnWithRegst(idbn, in_diff_regst);
      } else {
        CHECK(found_lbis.empty() || found_lbis.find(lbi) != found_lbis.end());
      }
    }
  });
}

void NormalBackwardCompTaskNode::BindInRegst() {
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      if (GetRelatedFwTaskNode()) {
        cur_node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
      }
    }
  });
}

void NormalBackwardCompTaskNode::BindModelDiffRegst() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetSoleConsumedRegst("data_tmp");
  std::shared_ptr<RegstDesc> bw_buf_regst = GetProducedRegst("bw_buf");
  std::shared_ptr<RegstDesc> const_model_regst = GetSoleConsumedRegst("const_model");
  std::shared_ptr<RegstDesc> model_regst = GetSoleConsumedRegst("model");
  std::shared_ptr<RegstDesc> const_buf_regst = GetSoleConsumedRegst("const_buf");
  std::shared_ptr<RegstDesc> model_diff_regst = GetProducedRegst("model_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->BindBnsWithRegst(&Operator::data_tmp_bns, data_tmp_regst);
    node->AddBnToRegstAndBindIt(&Operator::bw_buf_bns, bw_buf_regst);
    node->BindBnsWithRegst(&Operator::const_model_bns, const_model_regst);
    node->BindBnsWithRegst(&Operator::const_buf_bns, const_buf_regst);
    node->BindBnsWithRegst(&Operator::model_diff_bns, model_diff_regst);
    node->BindBnsWithRegst(&Operator::model_bns, model_regst);
  });
}

void NormalBackwardCompTaskNode::InferBlobDescsInProducedRegsts() {
  if (GetRelatedFwTaskNode()) {
    std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
    for (std::shared_ptr<RegstDesc> regst : GetConsumedRegst("in")) {
      in_diff_regst->CopyBlobDescWithoutAddLbi(regst.get());
    }

    std::shared_ptr<RegstDesc> md_diff_regst = GetProducedRegst("model_diff");
    if (md_diff_regst) {
      md_diff_regst->CopyBlobDescFrom(GetSoleConsumedRegst("model").get());
      if (type_name4model_related_logical_node_ == "MdDiffAcc") {
        md_diff_regst->ForEachLbi([md_diff_regst](const LogicalBlobId& lbi) {
          BlobDesc* blob_desc = md_diff_regst->MutBlobDesc(lbi);
          blob_desc->SetHasField<FieldKey::kDim0ValidNum>(true);
          blob_desc->mut_dim0_inner_shape() = Shape({1, blob_desc->shape().At(0)});
        });
      }
    }

    std::shared_ptr<RegstDesc> activation_diff_regst = GetProducedRegst("activation_diff");
    activation_diff_regst->CopyBlobDescWithoutAddLbi(GetSoleConsumedRegst("activation").get());
    activation_diff_regst->CopyBlobDescWithoutAddLbi(GetSoleConsumedRegst("out").get());
  } else {
    mut_exec_gph().SoleNode()->InferDiffBlobDescsWithoutFwNode(parallel_ctx());
  }
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) {
    node->FixInDiffBlobDescs(parallel_ctx());
    node->InferBwBufBlobDescs(parallel_ctx());
  });
}

CompTaskNode* NormalBackwardCompTaskNode::GetRelatedFwTaskNode() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* fw_node = edge->src_node();
    if (IsForwardTaskType(fw_node->GetTaskType())) { return static_cast<CompTaskNode*>(fw_node); }
  }
  return nullptr;
}

void NormalBackwardCompTaskNode::FixPackedBlobDescOfProducedRegst() {
  std::shared_ptr<RegstDesc> model_diff_regst = GetProducedRegst("model_diff");
  if (model_diff_regst == nullptr) { return; }
  CHECK(model_diff_regst->IsLocked());
  Shape& shape = model_diff_regst->MutBlobDesc(GenPackedLbi())->mut_shape();
  shape = Shape({static_cast<int64_t>(RoundUp(shape.elem_cnt(), parallel_ctx()->parallel_num()))});
}

void NormalBackwardCompTaskNode::RmUselessConsumeRelationshipToFw() {
  bool need_in_blob = false;
  bool need_out_blob = false;
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    if (node->in_edges().empty()) {
      need_in_blob = need_in_blob || node->op()->NeedInBlobWhenBackwardIf();
    }
    if (node->out_edges().empty()) {
      need_out_blob = need_out_blob || node->op()->NeedOutBlobWhenBackwardIf();
    }
  });
  if (need_in_blob == false) { EraseConsumedRegstsByName("in"); }
  if (need_out_blob == false) { EraseConsumedRegstsByName("out"); }
}

void NormalBackwardCompTaskNode::InferProducedDataRegstTimeShape() {
  const std::list<std::shared_ptr<RegstDesc>>& out_diff_regsts = GetConsumedRegst("out_diff");
  CHECK(!out_diff_regsts.empty());
  const std::shared_ptr<Shape>& out_diff_time_shape =
      out_diff_regsts.front()->data_regst_time_shape();
  for (const auto& regst : out_diff_regsts) {
    CHECK(*out_diff_time_shape == *(regst->data_regst_time_shape()));
  }

  ForEachProducedDataRegst([out_diff_time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = out_diff_time_shape;
  });
}

}  // namespace oneflow
