#include "oneflow/core/job_completer/clone_grad.h"

namespace oneflow {

void GenerateCloneGradOpIfNeed(
    const OpNode& op_node, std::vector<OperatorConf>* op_confs,
    const HashMap<std::string, HashMap<std::string, LogicalBlobId>>& op_name2ibn2in_diff_lbi,
    HashMap<LogicalBlobId, LogicalBlobId>* lbi2out_diff_lbi) {
  HashMap<LogicalBlobId, std::vector<LogicalBlobId>> in_lbi2in_diff_lbis;
  op_node.ForEachNodeOnOutEdge([&](OpNode* out_node) {
    for (const auto& ibn : out_node->op().input_bns()) {
      const auto& op_iter = op_name2ibn2in_diff_lbi.find(out_node->op().op_name());
      if (op_iter == op_name2ibn2in_diff_lbi.end()) { continue; }
      const auto& iter = op_iter->second.find(ibn);
      if (iter == op_iter->second.end()) { continue; }
      in_lbi2in_diff_lbis[out_node->op().BnInOp2Lbi(ibn)].push_back(iter->second);
    }
  });
  for (const auto& obn : op_node.op().output_bns()) {
    const LogicalBlobId& lbi = op_node.op().BnInOp2Lbi(obn);
    LogicalBlobId diff_lbi;
    const auto& in_diff_lbis = in_lbi2in_diff_lbis[lbi];
    if (in_diff_lbis.empty()) { continue; }
    if (in_diff_lbis.size() == 1) {
      diff_lbi = in_diff_lbis.at(0);
    } else if (in_diff_lbis.size() > 1) {
      OperatorConf add_op;
      add_op.set_name(op_node.op().op_name() + "_clone_grad_" + NewUniqueId());
      AddOpConf* add_op_conf = add_op.mutable_add_conf();
      add_op_conf->set_out("out");
      for (const auto& in_diff_lbi : in_diff_lbis) {
        add_op_conf->add_in(GenLogicalBlobName(in_diff_lbi));
      }
      op_confs->push_back(add_op);
      diff_lbi.set_op_name(add_op.name());
      diff_lbi.set_blob_name("out");
    }
    lbi2out_diff_lbi->emplace(lbi, diff_lbi);
  }
}

}  // namespace oneflow
