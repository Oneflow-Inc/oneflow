/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/job_rewriter/clone_grad.h"
#include "oneflow/core/framework/framework.h"

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
    std::vector<LogicalBlobId> lbis_to_add = std::move(out_oba2in_diff_lbis[oba]);
    if (lbis_to_add.empty()) { continue; }
    bool need_add = lbis_to_add.size() > 1;
    while (lbis_to_add.size() != 1) {
      const size_t add_n_op_max_input_num = 8;
      user_op::UserOpConfWrapperBuilder add_op_builder(op_node.op().op_name() + "_clone_grad_"
                                                       + NewUniqueId());
      add_op_builder.Op("add_n");
      const size_t start = lbis_to_add.size() >= add_n_op_max_input_num
                               ? lbis_to_add.size() - add_n_op_max_input_num
                               : 0;
      for (size_t i = start; i < lbis_to_add.size(); ++i) {
        add_op_builder.Input("in", GenLogicalBlobName(lbis_to_add.at(i)));
      }
      lbis_to_add.resize(start);
      const auto add_op =
          add_op_builder.Output("out")
              .ScopeSymbolId(job_builder->OpConf4OpName(lbi.op_name()).scope_symbol_id())
              .Build();
      job_builder->AddOps(job_builder->ParallelConf4Lbi(lbi), {add_op.op_conf()});
      lbis_to_add.push_back(GenLogicalBlobId(add_op.output("out", 0)));
    }
    if (need_add) { CHECK(out_oba2clone_bw_add_out_lbi->emplace(oba, lbis_to_add.front()).second); }
    out_oba2out_diff_lbi->emplace(oba, lbis_to_add.front());
  }
}

}  // namespace oneflow
