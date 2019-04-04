#include "oneflow/core/job_completer/clone_grad.h"

namespace oneflow {

void GenerateCloneGradOpIfNeed(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const HashMap<LogicalBlobId, HashMap<std::string, LogicalBlobId>>& lbi2op_name2in_diff_lbi,
    HashMap<LogicalBlobId, LogicalBlobId>* lbi2out_diff_lbi) {
  for (const auto& obn : op.output_bns()) {
    const LogicalBlobId& lbi = op.BnInOp2Lbi(obn);
    LogicalBlobId diff_lbi;
    if (lbi2op_name2in_diff_lbi.find(lbi) == lbi2op_name2in_diff_lbi.end()) { continue; }
    const auto& op_name2in_diff_lbi = lbi2op_name2in_diff_lbi.at(lbi);
    if (op_name2in_diff_lbi.size() == 1) {
      diff_lbi = op_name2in_diff_lbi.begin()->second;
    } else {
      OperatorConf add_op;
      add_op.set_name(op.op_name() + "_clone_grad_" + obn);
      AddOpConf* add_op_conf = add_op.mutable_add_conf();
      add_op_conf->set_out("out");
      for (const auto& pair : op_name2in_diff_lbi) {
        add_op_conf->add_in(GenLogicalBlobName(pair.second));
      }
      op_confs->push_back(add_op);
      diff_lbi.set_op_name(add_op.name());
      diff_lbi.set_blob_name("out");
    }
    lbi2out_diff_lbi->emplace(lbi, diff_lbi);
  }
}

}  // namespace oneflow
