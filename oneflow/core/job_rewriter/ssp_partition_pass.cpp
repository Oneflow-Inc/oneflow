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
                                 std::function<Maybe<int64_t>(OpNode*)>* ReverseDepth4OpNode);

Maybe<void> FuseTrainableFwOpsStartingFromVar(
    const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
    const std::function<Maybe<int64_t>(OpNode*)>& ReverseDepth4OpNode,
    HashMap<OpNode*, OpNode*>* trainable_fw_op2backbone_op);
Maybe<void> FuseUnTrainableFwOpToNearestLastRanTrainableOp(
    const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
    const std::function<Maybe<int64_t>(OpNode*)>& ReverseDepth4OpNode,
    HashMap<OpNode*, OpNode*>* fused_untrainable_op2trainable_op);

Maybe<void> GetBackboneOp2FusedOps(
    const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
    HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>>* backbone_op2fused_ops) {
  // Gets reverse depth for trainable forward op
  std::function<Maybe<int64_t>(OpNode*)> ReverseDepth4OpNode;
  JUST(GetNode2ReverseDepth(op_graph, trainable_fw_ops, &ReverseDepth4OpNode));
  // Fuses trainable forward op to backbone op
  HashMap<OpNode*, OpNode*> trainable_fw_op2backbone_op;
  JUST(FuseTrainableFwOpsStartingFromVar(op_graph, trainable_fw_ops, ReverseDepth4OpNode,
                                         &trainable_fw_op2backbone_op));
  // Fuses untrainable op to trainable forward op
  HashMap<OpNode*, OpNode*> fused_untrainable_op2trainable_op;
  JUST(FuseUnTrainableFwOpToNearestLastRanTrainableOp(
      op_graph, trainable_fw_ops, ReverseDepth4OpNode, &fused_untrainable_op2trainable_op));
  const auto& AppendToBackbone = [&](OpNode* node, OpNode* backbone_op) {
    auto& ptr = (*backbone_op2fused_ops)[backbone_op];
    if (!ptr) { ptr.reset(new std::vector<OpNode*>()); }
    ptr->push_back(node);
  };
  for (const auto& pair : trainable_fw_op2backbone_op) {
    AppendToBackbone(pair.first, pair.second);
  }
  for (const auto& pair : fused_untrainable_op2trainable_op) {
    AppendToBackbone(pair.first, trainable_fw_op2backbone_op.at(pair.second));
  }
  return Maybe<void>::Ok();
}

Maybe<void> FuseUnTrainableFwOpToNearestLastRanTrainableOp(
    const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
    const std::function<Maybe<int64_t>(OpNode*)>& ReverseDepth4OpNode,
    HashMap<OpNode*, OpNode*>* fused_untrainable_op2trainable_op) {
  const auto& GetFusedTrainableOpNode = [&](OpNode* op_node) -> OpNode* {
    if (trainable_fw_ops.count(op_node) > 0) { return op_node; }
    const auto& iter = fused_untrainable_op2trainable_op->find(op_node);
    if (iter == fused_untrainable_op2trainable_op->end()) { return nullptr; }
    return iter->second;
  };
  const auto& GetLastRanFusedTrainableOpNode = [&](OpNode* untrainable_op) -> OpNode* {
    CHECK_EQ(trainable_fw_ops.count(untrainable_op), 0);
    int64_t reverse_depth = GetMaxVal<int64_t>();
    OpNode* last_run_trainable_op_node = nullptr;
    untrainable_op->ForEachNodeOnInOutEdge([&](OpNode* cur_node) {
      OpNode* cur_trainable_op_node = GetFusedTrainableOpNode(cur_node);
      if (cur_trainable_op_node == nullptr) { return; /* Bfs will see this node later. */ }
      int64_t cur_reverse_depth = CHECK_JUST(ReverseDepth4OpNode(cur_trainable_op_node));
      if (cur_reverse_depth < reverse_depth) {
        cur_reverse_depth = reverse_depth;
        last_run_trainable_op_node = cur_trainable_op_node;
      }
    });
    // Bfs will ensure that there is at least one op_node will be found.
    CHECK_NOTNULL(last_run_trainable_op_node);
    return last_run_trainable_op_node;
  };
  std::list<OpNode*> starts(trainable_fw_ops.begin(), trainable_fw_ops.end());
  const auto& ForEachConnected = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnInOutEdge([&](OpNode* in_or_out_node) {
      if (trainable_fw_ops.count(in_or_out_node) == 0) { Handler(in_or_out_node); }
    });
  };
  op_graph.BfsForEachNode(starts, ForEachConnected, [&](OpNode* op_node) {
    if (trainable_fw_ops.count(op_node)) { return; }
    (*fused_untrainable_op2trainable_op)[op_node] = GetLastRanFusedTrainableOpNode(op_node);
  });
  return Maybe<void>::Ok();
}

Maybe<void> FuseTrainableFwOpsStartingFromVar(
    const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
    const std::function<Maybe<int64_t>(OpNode*)>& ReverseDepth4OpNode,
    HashMap<OpNode*, OpNode*>* trainable_fw_op2backbone_op) {
  const auto& ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (trainable_fw_ops.count(out_node) > 0) { Handler(out_node); }
    });
  };
  for (OpNode* fw_node : trainable_fw_ops) { (*trainable_fw_op2backbone_op)[fw_node] = fw_node; }
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
      ForEachOut(node, [&](OpNode* out_node) {
        int64_t out_node_reverse_depth = CHECK_JUST(ReverseDepth4OpNode(out_node));
        if (out_node_reverse_depth > max_reverse_depth) {
          max_reverse_depth = out_node_reverse_depth;
          max_reverse_depth_op_node = out_node;
        }
      });
      // Fuses variable op to earlier ran forward op which has max reverse depth
      if (max_reverse_depth_op_node != nullptr) { Handler(max_reverse_depth_op_node); }
    };
    std::list<OpNode*> fused;
    // Fuses nodes starting from variable
    op_graph.BfsForEachNode({var_node}, ForEachNext, [&](OpNode* node) { fused.push_back(node); });
    CHECK(!fused.empty());
    OpNode* last = fused.back();
    for (OpNode* node : fused) { (*trainable_fw_op2backbone_op)[node] = last; }
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetNode2ReverseDepth(const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
                                 std::function<Maybe<int64_t>(OpNode*)>* ReverseDepth4OpNode) {
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
  auto node2reverse_depth = std::make_shared<HashMap<OpNode*, int64_t>>();
  int64_t reverse_depth = 1;
  op_graph.BfsForEachNode(starts, ForEachNext,
                          [&](OpNode* node) { (*node2reverse_depth)[node] = reverse_depth; });
  *ReverseDepth4OpNode = [node2reverse_depth](OpNode* op_node) -> Maybe<int64_t> {
    const auto& iter = node2reverse_depth->find(op_node);
    CHECK_OR_RETURN(iter != node2reverse_depth->end());
    return iter->second;
  };
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
