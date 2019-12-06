#include "oneflow/core/job_completer/clone_grad.h"

namespace oneflow {

void GenerateCloneGradOpIfNeed(const OpNode& op_node, JobBuilder* job_builder,
                               const HashMap<OpBlobArg, LogicalBlobId>& in_oba2in_diff_lbi,
                               HashMap<OpBlobArg, LogicalBlobId>* out_oba2out_diff_lbi,
                               HashMap<OpBlobArg, LogicalBlobId>* out_oba2clone_bw_add_out_lbi) {
  HashMap<LogicalBlobId, OpBlobArg> out_lbi2out_oba;
  for (const auto& obn : op_node.op().output_bns()) {
    out_lbi2out_oba[op_node.op().BnInOp2Lbi(obn)] = GenOpBlobArg(op_node.op().op_name(), obn);
  }
  HashMap<OpBlobArg, std::vector<LogicalBlobId>> out_oba2in_diff_lbis;
  op_node.ForEachNodeOnOutEdge([&](OpNode* out_node) {
    for (const auto& ibn : out_node->op().input_bns()) {
      const auto& oba_it = out_lbi2out_oba.find(out_node->op().BnInOp2Lbi(ibn));
      if (oba_it == out_lbi2out_oba.end()) { continue; }
      const auto& in_diff_lbi_it =
          in_oba2in_diff_lbi.find(GenOpBlobArg(out_node->op().op_name(), ibn));
      if (in_diff_lbi_it == in_oba2in_diff_lbi.end()) { continue; }
      out_oba2in_diff_lbis[oba_it->second].push_back(in_diff_lbi_it->second);
    }
  });
  for (const auto& obn : op_node.op().output_bns()) {
    const OpBlobArg& oba = GenOpBlobArg(op_node.op().op_name(), obn);
    const LogicalBlobId& lbi = op_node.op().BnInOp2Lbi(obn);
    LogicalBlobId diff_lbi;
    const auto& in_diff_lbis = out_oba2in_diff_lbis[oba];
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
      job_builder->AddOps(job_builder->ParallelConf4Lbi(lbi), {add_op});
      diff_lbi.set_op_name(add_op.name());
      diff_lbi.set_blob_name("out");
      CHECK(out_oba2clone_bw_add_out_lbi->emplace(oba, diff_lbi).second);
    } else {
      UNIMPLEMENTED();
    }
    out_oba2out_diff_lbi->emplace(oba, diff_lbi);
  }
}

}  // namespace oneflow
