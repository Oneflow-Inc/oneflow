#include "oneflow/core/job/autograd.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_conf_builder.h"

namespace oneflow {

namespace {

void GetLossOpNodes(const OpGraph& op_graph, std::list<OpNode*>* loss_op_nodes) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().IsLossOp()) { loss_op_nodes->push_back(op_node); }
  });
}

void ForEachInOpNode(OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
  op_node->ForEachNodeOnInEdge([&](OpNode* in_node) {
    if (in_node->has_in_diff() || in_node->has_model_diff()) { Handler(in_node); }
  });
}

void ForEachOutOpNode(OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
  // TODO: fix has_actual_out_diff
  op_node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
    if (!out_node->op().IsLossOp()) { Handler(out_node); }
  });
}

}  // namespace

JobConf1 AutoGrad(const JobDesc& job_desc) {
  JobConf1 job_conf(job_desc.job_conf());
  OpGraph op_graph(&job_desc);
  std::list<OpNode*> loss_op_nodes;
  GetLossOpNodes(op_graph, &loss_op_nodes);
  JobConfBuilder job_conf_builder(&job_conf);
  HashMap<LogicalBlobId, LogicalBlobId> lbi2grad_lbi;

  auto GenerateBackwardOpConf = [&](OpNode* op_node) {
    auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
      const auto& input_bns = op_node->op().input_bns();
      const auto& output_bns = op_node->op().output_bns();
      const auto& lbi = op_node->op().BnInOp2Lbi(bn);
      if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
        return &lbi2grad_lbi.at(lbi);
      } else if (std::find(output_bns.begin(), output_bns.end(), bn) != output_bns.end()) {
        if (lbi2grad_lbi.find(lbi) == lbi2grad_lbi.end()) { return nullptr; }
        return &lbi2grad_lbi.at(lbi);
      } else {
        UNIMPLEMENTED();
      }
    };
    op_node->op().GenerateBackwardOpConfIf(
        &job_conf_builder, op_node->parallel_desc().parallel_conf(), DiffLbi4BnInOp);
  };
  op_graph.TopoForEachNode(loss_op_nodes, &ForEachOutOpNode, &ForEachInOpNode,
                           GenerateBackwardOpConf);
  return job_conf;
}

}  // namespace oneflow
