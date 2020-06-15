#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

std::function<OperatorConf*(const std::string&)> MakeMutableOperatorConf4OpName(Job* job) {
  auto op_name2op_conf = std::make_shared<HashMap<std::string, OperatorConf*>>();
  FOR_RANGE(int, idx, 0, job->net().op_size()) {
    OperatorConf* op_conf = job->mutable_net()->mutable_op(idx);
    CHECK(op_name2op_conf->emplace(op_conf->name(), op_conf).second);
  }
  return [op_name2op_conf](const std::string& op_name) { return op_name2op_conf->at(op_name); };
}

void AddIdentityOp(const std::string& op_name, Job* job, const HashSet<LogicalBlobId>& input_lbis,
                   HashMap<LogicalBlobId, LogicalBlobId>* old_lbi2new_lbi,
                   HashMap<LogicalBlobId, std::string>* old_lbi2new_obn,
                   const ParallelConf& parallel_conf) {
  // add tuple identity op
  OperatorConf* tuple_identity_op = job->mutable_net()->add_op();
  tuple_identity_op->set_name(op_name);
  TupleIdentityOpConf* tuple_identity_op_conf = tuple_identity_op->mutable_tuple_identity_conf();
  int32_t idx = 0;
  for (const LogicalBlobId& lbi : input_lbis) {
    const std::string& obn = GenRepeatedBn("out", idx++);
    const std::string& blob_name = obn;
    {
      LogicalBlobId output_lbi;
      output_lbi.set_op_name(tuple_identity_op->name());
      output_lbi.set_blob_name(blob_name);
      CHECK(old_lbi2new_lbi->emplace(lbi, output_lbi).second);
      CHECK(old_lbi2new_obn->emplace(lbi, obn).second);
    }
    tuple_identity_op_conf->add_in(lbi.op_name() + "/" + lbi.blob_name());
    tuple_identity_op_conf->add_out(blob_name);
  }
  // add placement of tuple identity op
  PlacementGroup* p_group = job->mutable_placement()->add_placement_group();
  *(p_group->mutable_op_set()->add_op_name()) = tuple_identity_op->name();
  *(p_group->mutable_parallel_conf()) = parallel_conf;
}

void AddIdentityOpAndReconnect(
    const std::string& op_name_prefix, Job* job, const std::vector<OpEdge*>& op_edges,
    const std::function<OperatorConf*(const std::string&)>& MutOperatorConf4OpName,
    const ParallelConf& parallel_conf) {
  // add identity op
  HashSet<LogicalBlobId> lbis;
  for (OpEdge* edge : op_edges) { lbis.insert(edge->lbis().begin(), edge->lbis().end()); }
  HashMap<LogicalBlobId, LogicalBlobId> old_lbi2new_lbi;
  HashMap<LogicalBlobId, std::string> old_lbi2new_obn;
  const auto& identity_op_name = op_name_prefix + NewUniqueId();
  AddIdentityOp(identity_op_name, job, lbis, &old_lbi2new_lbi, &old_lbi2new_obn, parallel_conf);
  // reconnect to identity op
  HashMap<LogicalBlobId, SbpParallel> old_lbi2sbp_parallel;
  for (OpEdge* edge : op_edges) {
    OperatorConf* op_conf = MutOperatorConf4OpName(edge->dst_node()->op().op_name());
    PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
    for (const LogicalBlobId& lbi : edge->lbis()) {
      std::string lbn_check = GenLogicalBlobName(lbi);
      std::string identity_out_lbn = GenLogicalBlobName(old_lbi2new_lbi.at(lbi));
      for (const std::string& ibn : edge->lbi2ibns().at(lbi)) {
        ReplaceInputLbnInOpCustomizedConf(op_type_conf, ibn, lbn_check, identity_out_lbn);
        const auto& sbp_parallel = edge->dst_node()->SbpParallel4BnInOp(ibn);
        const auto& sbp_iter = old_lbi2sbp_parallel.find(lbi);
        if (sbp_iter == old_lbi2sbp_parallel.end()) {
          old_lbi2sbp_parallel[lbi] = sbp_parallel;
        } else {
          CHECK(sbp_iter->second == sbp_parallel);
        }
      }
    }
  }
  // set sbp conf for tuple_identity_op
  CHECK_EQ(old_lbi2new_obn.size(), old_lbi2sbp_parallel.size());
  auto* sbp_sig_conf_map =
      job->mutable_job_parallel_view_conf()->mutable_op_name2sbp_signature_conf();
  auto* sbp_parallel_map = (*sbp_sig_conf_map)[identity_op_name].mutable_bn_in_op2sbp_parallel();
  for (const auto& pair : old_lbi2new_obn) {
    (*sbp_parallel_map)[pair.second] = old_lbi2sbp_parallel.at(pair.first);
  }
}

std::function<bool(OpNode*)> MakePredicatorIsReachableFromAnyVariableOps(const OpGraph& op_graph) {
  auto vars_and_decendants = std::make_shared<HashSet<OpNode*>>();
  GetVariableOpNodesAndDescendants(op_graph, vars_and_decendants.get());
  return [vars_and_decendants](OpNode* op_node) -> bool {
    return vars_and_decendants->find(op_node) != vars_and_decendants->end();
  };
}

REGISTER_FUNCTION_CONFIG_DEF().Bool("enable_pseudo_chain_merge", false,
                                    "ties up chain headers unreachable from any variable ops");
class TieUpChainHeadersUnReachableFromAnyVariableOps final : public OpGraphPass {
  bool IsEnabled() const override {
    return GlobalJobDesc().IsTrain() && GlobalJobDesc().Bool("enable_pseudo_chain_merge");
  }

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const override {
    auto IsReachableFromAnyVariableOps = MakePredicatorIsReachableFromAnyVariableOps(op_graph);
    auto GetSourceNodesAndEdges = [&](const HashSet<OpNode*>& chain_nodes,
                                      std::vector<OpNode*>* source_nodes,
                                      std::vector<OpEdge*>* source_edges) {
      for (OpNode* node : chain_nodes) {
        for (OpEdge* edge : node->in_edges()) {
          if (chain_nodes.find(edge->src_node()) == chain_nodes.end()
              && IsReachableFromAnyVariableOps(edge->src_node()) == false) {
            source_edges->push_back(edge);
            source_nodes->push_back(node);
          }
        }
      }
    };
    auto MutOperatorConf4OpName = MakeMutableOperatorConf4OpName(job);
    auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job->placement());
    op_graph.ForEachChainFamily([&](const HashSet<OpNode*>& chain_nodes) {
      std::vector<OpNode*> source_nodes;
      std::vector<OpEdge*> source_edges;
      GetSourceNodesAndEdges(chain_nodes, &source_nodes, &source_edges);
      if (source_edges.size() <= 1) { return; }
      if (source_nodes.size() <= 1) { return; }
      // ignore small chain
      if (chain_nodes.size() - source_nodes.size() <= 2) { return; }
      AddIdentityOpAndReconnect("pseudo_chain_header_", job, source_edges, MutOperatorConf4OpName,
                                *ParallelConf4OpName(source_nodes.at(0)->op().op_name()));
    });
    return Maybe<void>::Ok();
  }
};

REGISTER_FUNCTION_PASS("TieUpChainHeadersUnReachableFromAnyVariableOps",
                       TieUpChainHeadersUnReachableFromAnyVariableOps);

}  // namespace

}  // namespace oneflow
