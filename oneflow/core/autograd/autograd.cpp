#include "oneflow/core/autograd/autograd.h"
#include "oneflow/core/job/job_conf_builder.h"
#include "oneflow/core/autograd/clone_grad.h"

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

std::function<bool(const LogicalBlobId&, const std::string&)> MakePredicatorHasDiff4LbiOpName(
    const OpGraph& op_graph, const std::function<bool(OpNode*)>& NeedBackwardOp) {
  auto lbis2ops_with_in_diff = std::make_shared<HashMap<LogicalBlobId, HashSet<std::string>>>();
  op_graph.ForEachEdge([&](OpEdge* edge) {
    if (NeedBackwardOp(edge->src_node()) && NeedBackwardOp(edge->dst_node())) {
      for (const auto& lbi : edge->lbis()) {
        if (edge->src_node()->op().HasOutDiff4Lbi(lbi)) {
          (*lbis2ops_with_in_diff)[lbi].emplace(edge->dst_node()->op().op_name());
        }
      }
    }
  });
  return [lbis2ops_with_in_diff](const LogicalBlobId& lbi, const std::string& op_name) {
    if (lbis2ops_with_in_diff->find(lbi) == lbis2ops_with_in_diff->end()) { return false; }
    const auto& op_names = lbis2ops_with_in_diff->at(lbi);
    return op_names.find(op_name) != op_names.end();
  };
}

}  // namespace

void GenerateBackwardOpConfWrapperStruct::Call(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) const {
  if (func_) {
    (*func_)(op, op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
  } else if (naive_func_) {
    (*naive_func_)(op, op_confs, DiffLbi4BnInOp);
  } else {
    UNIMPLEMENTED();
  }
}

void GenerateBackwardOpConfIf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  auto* obj = NewObj<GenerateBackwardOpConfWrapperStruct>(op.op_conf().op_type_case());
  obj->Call(op, op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
}

void AutoGrad(const OpGraph& op_graph, JobConf1* job_conf,
              HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  CHECK(lbi2diff_lbi->empty());
  auto NeedBackwardOp = MakePredicatorNeedBackwardOp(op_graph);
  std::list<OpNode*> loss_nodes;
  GetLossOpNodes(op_graph, &loss_nodes);
  for (OpNode* loss_node : loss_nodes) { CHECK(NeedBackwardOp(loss_node)); }

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
  auto HasDiff4LbiOpName = MakePredicatorHasDiff4LbiOpName(op_graph, NeedBackwardOp);
  HashMap<LogicalBlobId, HashMap<std::string, LogicalBlobId>> lbi2op_name2in_diff_lbi;
  HashMap<LogicalBlobId, LogicalBlobId>* lbi2out_diff_lbi = lbi2diff_lbi;
  JobConfBuilder job_conf_builder(job_conf);
  op_graph.TopoForEachNode(loss_nodes, ForEachOutNode, ForEachInNode, [&](OpNode* op_node) {
    const auto& op_name = op_node->op().op_name();
    auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
      const auto& input_bns = op_node->op().input_bns();
      const auto& output_bns = op_node->op().output_bns();
      const auto& lbi = op_node->op().BnInOp2Lbi(bn);
      if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
        return HasDiff4LbiOpName(lbi, op_name) ? &lbi2op_name2in_diff_lbi[lbi][op_name] : nullptr;
      } else if (std::find(output_bns.begin(), output_bns.end(), bn) != output_bns.end()) {
        if (lbi2out_diff_lbi->find(lbi) == lbi2out_diff_lbi->end()) { return nullptr; }
        return &lbi2out_diff_lbi->at(lbi);
      } else {
        UNIMPLEMENTED();
      }
    };
    auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
      return op_graph.GetLogicalBlobDesc(op_node->op().BnInOp2Lbi(bn));
    };
    std::vector<OperatorConf> ops;
    GenerateCloneGradOpIfNeed(op_node->op(), &ops, lbi2op_name2in_diff_lbi, lbi2out_diff_lbi);
    GenerateBackwardOpConfIf(op_node->op(), &ops, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
    job_conf_builder.AddOps(op_node->parallel_desc().parallel_conf(), ops);
  });
}

void AddTotalLossInstanceNumOpConf(const OpGraph& op_graph, JobConf1* job_conf,
                                   LogicalBlobId* total_loss_instance_num_lbi) {
  std::list<OpNode*> loss_nodes;
  GetLossOpNodes(op_graph, &loss_nodes);
  OperatorConf op_conf;
  op_conf.set_name("system-autograd-total_loss_instance_num");
  TotalLossInstanceNumOpConf* conf = op_conf.mutable_total_loss_instance_num_conf();
  std::vector<LogicalBlobId> loss_instance_num_lbis;
  for (const OpNode* op_node : loss_nodes) {
    LogicalBlobId lbi;
    lbi.set_op_name(op_node->op().op_name());
    lbi.set_blob_name("loss_instance_num");
    conf->add_in(GenLogicalBlobName(lbi));
  }
  conf->set_out("out");

  ParallelConf parallel_conf;
  parallel_conf.set_policy(kDataParallel);
  parallel_conf.add_device_name("0:cpu:0");
  JobConfBuilder(job_conf).AddOps(parallel_conf, {op_conf});

  total_loss_instance_num_lbi->set_op_name(op_conf.name());
  total_loss_instance_num_lbi->set_blob_name("out");
}

}  // namespace oneflow
