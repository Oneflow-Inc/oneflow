#include "oneflow/core/job/autograd.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_conf_builder.h"

namespace oneflow {

namespace {

void GetVariableOpNodesAndDescendants(const OpGraph& op_graph, HashSet<OpNode*>* op_nodes) {
  std::list<OpNode*> starts;
  op_graph.ForEachNode([&](OpNode* op_node) {
    const auto& op_conf = op_node->op().op_conf();
    if (op_conf.has_variable_conf()) { starts.push_back(op_node); }
  });
  op_graph.BfsForEachNode(starts, &OpNode::ForEachNodeOnOutEdge,
                          [&](OpNode* op_node) { op_nodes->emplace(op_node); });
}

void GetLossOpNodes(const OpGraph& op_graph, std::list<OpNode*>* loss_op_nodes) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().IsLossOp()) { loss_op_nodes->push_back(op_node); }
  });
}

void GetLossOpNodesAndAscendants(const OpGraph& op_graph, HashSet<OpNode*>* op_nodes) {
  std::list<OpNode*> starts;
  GetLossOpNodes(op_graph, &starts);
  op_graph.BfsForEachNode(starts, &OpNode::ForEachNodeOnInEdge,
                          [&](OpNode* op_node) { op_nodes->emplace(op_node); });
}

std::function<bool(OpNode*)> MakePredicatorNeedBackwardOp(const OpGraph& op_graph) {
  auto variable_op_nodes_and_descendants = std::make_shared<HashSet<OpNode*>>();
  GetVariableOpNodesAndDescendants(op_graph, variable_op_nodes_and_descendants.get());
  auto loss_op_nodes_and_ascendants = std::make_shared<HashSet<OpNode*>>();
  GetLossOpNodesAndAscendants(op_graph, loss_op_nodes_and_ascendants.get());
  return [variable_op_nodes_and_descendants, loss_op_nodes_and_ascendants](OpNode* op_node) {
    return variable_op_nodes_and_descendants->find(op_node)
               != variable_op_nodes_and_descendants->end()
           && loss_op_nodes_and_ascendants->find(op_node) != loss_op_nodes_and_ascendants->end();
  };
}

std::function<bool(const LogicalBlobId&)> MakePredicatorHasDiff4Lbi(
    const OpGraph& op_graph, const std::function<bool(OpNode*)>& NeedBackwardOp) {
  auto lbis_with_diff = std::make_shared<HashSet<LogicalBlobId>>();
  op_graph.ForEachEdge([&](OpEdge* edge) {
    if (NeedBackwardOp(edge->src_node())) {
      for (const auto& lbi : edge->lbis()) { lbis_with_diff->emplace(lbi); }
    }
  });
  return [lbis_with_diff](const LogicalBlobId& lbi) {
    return lbis_with_diff->find(lbi) != lbis_with_diff->end();
  };
}

}  // namespace

JobConf1 AutoGrad(const JobDesc& job_desc) {
  JobConf1 job_conf(job_desc.job_conf());
  OpGraph op_graph(&job_desc);
  auto NeedBackwardOp = MakePredicatorNeedBackwardOp(op_graph);
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (NeedBackwardOp(op_node)) { LOG(INFO) << "NeedBackwardOp: " << op_node->op().op_name(); }
  });
  std::list<OpNode*> start_nodes;
  {
    std::list<OpNode*> loss_nodes;
    GetLossOpNodes(op_graph, &loss_nodes);
    for (OpNode* loss_node : loss_nodes) {
      if (NeedBackwardOp(loss_node)) { start_nodes.push_back(loss_node); }
    }
  }

  auto ForEachInNode = [&](OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
    op_node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      if (NeedBackwardOp(in_node)) { Handler(in_node); }
    });
  };
  auto ForEachOutNode = [&](OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
    op_node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (NeedBackwardOp(out_node)) { Handler(out_node); }
    });
  };
  auto HasDiff4Lbi = MakePredicatorHasDiff4Lbi(op_graph, NeedBackwardOp);
  HashMap<LogicalBlobId, LogicalBlobId> lbi2grad_lbi;
  JobConfBuilder job_conf_builder(&job_conf);
  op_graph.TopoForEachNode(start_nodes, ForEachOutNode, ForEachInNode, [&](OpNode* op_node) {
    auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
      const auto& input_bns = op_node->op().input_bns();
      const auto& output_bns = op_node->op().output_bns();
      const auto& lbi = op_node->op().BnInOp2Lbi(bn);
      if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
        return HasDiff4Lbi(lbi) ? &lbi2grad_lbi[lbi] : nullptr;
      } else if (std::find(output_bns.begin(), output_bns.end(), bn) != output_bns.end()) {
        if (lbi2grad_lbi.find(lbi) == lbi2grad_lbi.end()) { return nullptr; }
        return &lbi2grad_lbi.at(lbi);
      } else {
        UNIMPLEMENTED();
      }
    };
    std::vector<OperatorConf> ops;
    op_node->op().GenerateBackwardOpConfIf(&ops, DiffLbi4BnInOp);
    job_conf_builder.AddOps(op_node->parallel_desc().parallel_conf(), ops);
    LOG(INFO) << op_node->op().op_name();
  });
  return job_conf;
}

}  // namespace oneflow
