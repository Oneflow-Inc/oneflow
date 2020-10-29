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
#include "oneflow/core/job_rewriter/op_graph_pass.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job/job_builder.h"

namespace oneflow {

namespace {

Maybe<void> ForEachSspParallelConf4TrainableFwOp(
    const OpGraph&, const std::function<void(const OpNode*, const ParallelConf&)>&);

class SspPartitionPass final : public OpGraphPass {
  bool IsEnabled() const override { return true; }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override {
    JUST(ForEachSspParallelConf4TrainableFwOp(
        op_graph, [&](const OpNode* op_node, const ParallelConf& parallel_conf) {
          job_builder->MutParallelConfOnlyOnce(op_node->op().op_name(), parallel_conf);
        }));
    return Maybe<void>::Ok();
  }
};

REGISTER_FUNCTION_PASS("SspPartition", SspPartitionPass);

Maybe<void> GetTrainableFwOp2Depth(const OpGraph&, HashMap<OpNode*, int64_t>* trainble_fw_op2depth);
Maybe<void> MakeGetterSspParallelConf4OpName(
    const OpGraph& op_graph, const HashMap<OpNode*, int64_t>& trainble_fw_op2depth,
    std::function<Maybe<const ParallelConf&>(const OpNode*)>* SspParallelConf4OpName);

Maybe<void> ForEachSspParallelConf4TrainableFwOp(
    const OpGraph& op_graph,
    const std::function<void(const OpNode*, const ParallelConf&)>& Handler) {
  HashMap<OpNode*, int64_t> trainble_fw_op2depth;
  JUST(GetTrainableFwOp2Depth(op_graph, &trainble_fw_op2depth));
  std::function<Maybe<const ParallelConf&>(const OpNode*)> SspParallelConf4OpName;
  JUST(MakeGetterSspParallelConf4OpName(op_graph, trainble_fw_op2depth, &SspParallelConf4OpName));
  for (const auto& pair : trainble_fw_op2depth) {
    Handler(pair.first, JUST(SspParallelConf4OpName(pair.first)));
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetTrainableFwOps(const OpGraph& op_graph, HashSet<OpNode*>* trainable_fw_ops);

Maybe<void> GetBackboneOp2FusedOps(const OpGraph& op_graph,
                                   const HashSet<OpNode*>& trainable_fw_ops,
                                   HashMap<OpNode*, std::vector<OpNode*>>* backbone_op2fused_ops);

Maybe<void> GetTrainableFwOp2Depth(const OpGraph& op_graph,
                                   HashMap<OpNode*, int64_t>* trainble_fw_op2depth) {
  HashSet<OpNode*> trainable_fw_ops;
  JUST(GetTrainableFwOps(op_graph, &trainable_fw_ops));
  HashMap<OpNode*, std::vector<OpNode*>> backbone_op2fused_ops;
  JUST(GetBackboneOp2FusedOps(op_graph, trainable_fw_ops, &backbone_op2fused_ops));
  std::list<OpNode*> starts;
  {
    const auto& ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](OpNode* in_node) {
        if (backbone_op2fused_ops.count(in_node) > 0) { Handler(in_node); }
      });
    };
    const auto& IsSourceNode = [&](OpNode* node) {
      size_t in_num = 0;
      ForEachIn(node, [&](OpNode*) { ++in_num; });
      return in_num == 0;
    };
    for (const auto& pair : backbone_op2fused_ops) {
      if (IsSourceNode(pair.first)) { starts.push_back(pair.first); }
    }
  }
  const auto& ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (backbone_op2fused_ops.count(out_node) > 0) { Handler(out_node); }
    });
  };
  int64_t depth = 1;
  op_graph.BfsForEachNode(starts, ForEachOut, [&](OpNode* node) {
    const auto& iter = backbone_op2fused_ops.find(node);
    CHECK(iter != backbone_op2fused_ops.end());
    for (OpNode* op_node : iter->second) { (*trainble_fw_op2depth)[op_node] = depth; }
    ++depth;
  });
  return Maybe<void>::Ok();
}

Maybe<void> GetTrainableFwOps(const OpGraph& op_graph, HashSet<OpNode*>* trainable_fw_ops) {
  std::function<bool(OpNode*)> NeedBackwardOp;
  JUST(MakePredicatorNeedBackwardOp(op_graph, &NeedBackwardOp));
  op_graph.ForEachNode([&](OpNode* node) {
    if (NeedBackwardOp(node)) { trainable_fw_ops->insert(node); }
  });
  return Maybe<void>::Ok();
}

Maybe<void> GetBackboneOp2FusedOps(const OpGraph& op_graph,
                                   const HashSet<OpNode*>& trainable_fw_ops,
                                   HashMap<OpNode*, std::vector<OpNode*>>* backbone_op2fused_ops) {
  TODO();
  return Maybe<void>::Ok();
}

Maybe<void> MakeGetterSspParallelConf4OpName(
    const OpGraph& op_graph, const HashMap<OpNode*, int64_t>& trainble_fw_op2depth,
    std::function<Maybe<const ParallelConf&>(const OpNode*)>* SspParallelConf4OpName) {
  TODO();
  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow
