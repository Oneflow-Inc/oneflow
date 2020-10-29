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
#include "oneflow/core/common/util.h"

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

Maybe<void> GetSequantialTrainableFwOps(
    const OpGraph&,
    std::vector<std::unique_ptr<std::vector<OpNode*>>>* sequantial_trainable_fw_ops);
Maybe<void> MakeGetterSspParallelConf4Depth(
    const std::vector<std::unique_ptr<std::vector<OpNode*>>>& sequantial_trainable_fw_ops,
    std::function<Maybe<const ParallelConf&>(int64_t)>* SspParallelConf4Depth);

Maybe<void> ForEachSspParallelConf4TrainableFwOp(
    const OpGraph& op_graph,
    const std::function<void(const OpNode*, const ParallelConf&)>& Handler) {
  std::vector<std::unique_ptr<std::vector<OpNode*>>> sequantial_trainable_fw_ops;
  JUST(GetSequantialTrainableFwOps(op_graph, &sequantial_trainable_fw_ops));
  std::function<Maybe<const ParallelConf&>(int64_t)> SspParallelConf4Depth;
  JUST(MakeGetterSspParallelConf4Depth(sequantial_trainable_fw_ops, &SspParallelConf4Depth));
  for (int64_t i = 0; i < sequantial_trainable_fw_ops.size(); ++i) {
    const auto& parallel_conf = JUST(SspParallelConf4Depth(i));
    for (OpNode* op_node : *sequantial_trainable_fw_ops.at(i)) { Handler(op_node, parallel_conf); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetTrainableFwOps(const OpGraph& op_graph, HashSet<OpNode*>* trainable_fw_ops);

Maybe<void> GetBackboneOp2FusedOps(
    const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
    HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>>* backbone_op2fused_ops);

Maybe<void> GetSequantialTrainableFwOps(
    const OpGraph& op_graph,
    std::vector<std::unique_ptr<std::vector<OpNode*>>>* sequantial_trainable_fw_ops) {
  HashSet<OpNode*> trainable_fw_ops;
  JUST(GetTrainableFwOps(op_graph, &trainable_fw_ops));
  HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>> backbone_op2fused_ops;
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
  op_graph.BfsForEachNode(starts, ForEachOut, [&](OpNode* op_node) {
    const auto& iter = backbone_op2fused_ops.find(op_node);
    CHECK(iter != backbone_op2fused_ops.end());
    sequantial_trainable_fw_ops->emplace_back(std::move(iter->second));
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

Maybe<void> GetNode2ReverseDepth(const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
                                 HashMap<OpNode*, int64_t>* node2reverse_depth);

Maybe<void> GetBackboneOp2FusedOps(
    const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
    HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>>* backbone_op2fused_ops) {
  HashMap<OpNode*, int64_t> node2reverse_depth;
  JUST(GetNode2ReverseDepth(op_graph, trainable_fw_ops, &node2reverse_depth));
  HashSet<OpNode*> remainder_fw_ops(trainable_fw_ops);
  for (OpNode* fw_node : trainable_fw_ops) {
    if (!fw_node->op().op_conf().has_variable_conf()) { continue; }
    OpNode* var_node = fw_node;
    const auto& ForEachNext = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      // It can not be applied to distribute_clone and distribute_split
      CHECK(!node->op().op_conf().has_distribute_clone_conf());
      CHECK(!node->op().op_conf().has_distribute_split_conf());
      // search until finding a multi-input op
      if (node->in_edges().size() > 1) { return; }
      int64_t max_reverse_depth = 0;
      OpNode* max_reverse_depth_op_node = nullptr;
      node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
        int64_t out_node_reverse_depth = node2reverse_depth[out_node];
        if (out_node_reverse_depth > max_reverse_depth) {
          max_reverse_depth = out_node_reverse_depth;
          max_reverse_depth_op_node = out_node;
        }
      });
      // Fuses variable op to earlier ran forward op which has max reverse depth
      if (max_reverse_depth_op_node != nullptr) { Handler(max_reverse_depth_op_node); }
    };
    auto fused_vec = std::make_unique<std::vector<OpNode*>>();
    // Fuses node start from variable
    op_graph.BfsForEachNode({var_node}, ForEachNext,
                            [&](OpNode* node) { fused_vec->push_back(node); });
    CHECK(!fused_vec->empty());
    for (size_t i = 0; i < fused_vec->size() - 1; ++i) { remainder_fw_ops.erase(fused_vec->at(i)); }
    (*backbone_op2fused_ops)[fused_vec->at(fused_vec->size() - 1)] = std::move(fused_vec);
  }
  // stain unfused for remainder forward ops
  for (OpNode* op_node : remainder_fw_ops) {
    (*backbone_op2fused_ops)[op_node].reset(new std::vector<OpNode*>{op_node});
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetNode2ReverseDepth(const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
                                 HashMap<OpNode*, int64_t>* node2reverse_depth) {
  std::list<OpNode*> starts;
  {
    const auto& ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
        if (trainable_fw_ops.count(out_node) > 0) { Handler(out_node); }
      });
    };
    const auto& IsSinkNode = [&](OpNode* node) {
      size_t out_num = 0;
      ForEachOut(node, [&](OpNode*) { ++out_num; });
      return out_num == 0;
    };
    for (OpNode* node : trainable_fw_ops) {
      if (IsSinkNode(node)) { starts.push_back(node); }
    }
  }
  const auto& ForEachNext = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      if (trainable_fw_ops.count(in_node) > 0) { Handler(in_node); }
    });
  };
  int64_t reverse_depth = 1;
  op_graph.BfsForEachNode(starts, ForEachNext,
                          [&](OpNode* node) { (*node2reverse_depth)[node] = reverse_depth; });
  return Maybe<void>::Ok();
}

Maybe<void> MakeGetterSspParallelConf4Depth(
    const std::vector<std::unique_ptr<std::vector<OpNode*>>>& sequantial_trainable_fw_ops,
    std::function<Maybe<const ParallelConf&>(int64_t)>* SspParallelConf4Depth) {
  TODO();
  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow
