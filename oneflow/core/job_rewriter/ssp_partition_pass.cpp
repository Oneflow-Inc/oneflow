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
    const OpGraph&, std::list<std::unique_ptr<std::vector<OpNode*>>>* sequantial_trainable_fw_ops);
Maybe<void> MakeGetterSspParallelConf4Depth(
    const std::list<std::unique_ptr<std::vector<OpNode*>>>& sequantial_trainable_fw_ops,
    std::function<Maybe<const ParallelConf&>(int64_t)>* SspParallelConf4Depth);

Maybe<void> ForEachSspParallelConf4TrainableFwOp(
    const OpGraph& op_graph,
    const std::function<void(const OpNode*, const ParallelConf&)>& Handler) {
  std::list<std::unique_ptr<std::vector<OpNode*>>> sequantial_trainable_fw_ops;
  JUST(GetSequantialTrainableFwOps(op_graph, &sequantial_trainable_fw_ops));
  std::function<Maybe<const ParallelConf&>(int64_t)> SspParallelConf4Depth;
  JUST(MakeGetterSspParallelConf4Depth(sequantial_trainable_fw_ops, &SspParallelConf4Depth));
  int64_t depth = 0;
  for (const auto& fused_vec : sequantial_trainable_fw_ops) {
    const auto& parallel_conf = JUST(SspParallelConf4Depth(depth));
    for (OpNode* op_node : *fused_vec) { Handler(op_node, parallel_conf); }
    ++depth;
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetTrainableFwOps(const OpGraph& op_graph, HashSet<OpNode*>* trainable_fw_ops);

Maybe<void> GetBackboneOp2FusedOps(
    const OpGraph& op_graph,
    HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>>* backbone_op2fused_ops);

Maybe<void> GetSequantialTrainableFwOps(
    const OpGraph& op_graph,
    std::list<std::unique_ptr<std::vector<OpNode*>>>* sequantial_trainable_fw_ops) {
  HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>> backbone_op2fused_ops;
  JUST(GetBackboneOp2FusedOps(op_graph, &backbone_op2fused_ops));
  std::list<OpNode*> starts;
  {
    const auto& ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
        if (backbone_op2fused_ops.count(out_node) > 0) { Handler(out_node); }
      });
    };
    const auto& IsSinkNode = [&](OpNode* node) {
      size_t out_num = 0;
      ForEachOut(node, [&](OpNode*) { ++out_num; });
      return out_num == 0;
    };
    for (const auto& pair : backbone_op2fused_ops) {
      if (IsSinkNode(pair.first)) { starts.push_back(pair.first); }
    }
  }
  const auto& ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      if (backbone_op2fused_ops.count(in_node) > 0) { Handler(in_node); }
    });
  };
  // Traverses reverserly
  op_graph.BfsForEachNode(starts, ForEachIn, [&](OpNode* op_node) {
    const auto& iter = backbone_op2fused_ops.find(op_node);
    CHECK(iter != backbone_op2fused_ops.end());
    sequantial_trainable_fw_ops->emplace_front(std::move(iter->second));
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

Maybe<void> GetBackBoneOps(const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
                           HashSet<OpNode*>* backbone_op_nodes);

Maybe<void> FuseOtherFwOpsToBackboneOps(const OpGraph& op_graph,
                                        const HashSet<OpNode*>& backbone_op_nodes,
                                        HashMap<OpNode*, OpNode*>* other_fw_op2backbone_op);

Maybe<void> GetBackboneOp2FusedOps(
    const OpGraph& op_graph,
    HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>>* backbone_op2fused_ops) {
  // Gets trainable forward ops.
  HashSet<OpNode*> trainable_fw_ops;
  JUST(GetTrainableFwOps(op_graph, &trainable_fw_ops));
  // Gets backbone ops.
  HashSet<OpNode*> backbone_op_nodes;
  JUST(GetBackBoneOps(op_graph, trainable_fw_ops, &backbone_op_nodes));
  // Fuses other forward ops to backbone ops.
  HashMap<OpNode*, OpNode*> other_fw_op2backbone_op;
  JUST(FuseOtherFwOpsToBackboneOps(op_graph, backbone_op_nodes, &other_fw_op2backbone_op));
  for (OpNode* backbone_op_node : backbone_op_nodes) {
    (*backbone_op2fused_ops)[backbone_op_node].reset(new std::vector<OpNode*>{backbone_op_node});
  }
  for (const auto& pair : other_fw_op2backbone_op) {
    (*backbone_op2fused_ops)[pair.second]->push_back(pair.first);
  }
  return Maybe<void>::Ok();
}

// subgraph trainable_fw_ops can be regarded as DAG whose source nodes are variable op nodes and
// whose sink nodes are loss op nodes.
//
// A op node is called backbone op node in trainable_fw_ops if:
//    a) it has two input in subgraph trainable_fw_ops;
//    b) or it has at least one backbone op as input
Maybe<void> GetBackBoneOps(const OpGraph& op_graph, const HashSet<OpNode*>& trainable_fw_ops,
                           HashSet<OpNode*>* backbone_op_nodes) {
  std::list<OpNode*> starts;
  {
    const auto& ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](OpNode* in_node) {
        if (trainable_fw_ops.count(in_node) > 0) { Handler(in_node); }
      });
    };
    const auto& GetInputSize = [&](OpNode* node) {
      size_t input_size = 0;
      ForEachIn(node, [&](OpNode*) { ++input_size; });
      return input_size;
    };
    for (OpNode* op_node : trainable_fw_ops) {
      if (GetInputSize(op_node) > 1) { starts.push_back(op_node); }
    }
  }
  const auto& ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (trainable_fw_ops.count(out_node) > 0) { Handler(out_node); }
    });
  };
  op_graph.BfsForEachNode(starts, ForEachOut,
                          [&](OpNode* node) { backbone_op_nodes->insert(node); });
  return Maybe<void>::Ok();
}

Maybe<void> BfsForEachBackboneOp(const OpGraph& op_graph, const HashSet<OpNode*>& backbone_op_nodes,
                                 const std::function<void(OpNode*)>& Handler);

Maybe<void> FuseOtherFwOpsToBackboneOps(const OpGraph& op_graph,
                                        const HashSet<OpNode*>& backbone_op_nodes,
                                        HashMap<OpNode*, OpNode*>* other_fw_op2backbone_op) {
  const auto& ForEachNextOther = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      if (backbone_op_nodes.count(in_node) > 0) { return; }
      // It's safe to update container other_fw_op2backbone_op when traversing.
      if (other_fw_op2backbone_op->count(in_node) > 0) { return; }
      // Traverses other nodes.
      Handler(in_node);
    });
  };
  const auto& DoEachBackboneOp = [&](OpNode* backbone_op_node) {
    op_graph.BfsForEachNode({backbone_op_node}, ForEachNextOther, [&](OpNode* other) {
      if (backbone_op_nodes.count(other) > 0) { return; }
      (*other_fw_op2backbone_op)[other] = backbone_op_node;
    });
  };
  JUST(BfsForEachBackboneOp(op_graph, backbone_op_nodes, DoEachBackboneOp));
  return Maybe<void>::Ok();
}

Maybe<void> BfsForEachBackboneOp(const OpGraph& op_graph, const HashSet<OpNode*>& backbone_op_nodes,
                                 const std::function<void(OpNode*)>& Handler) {
  std::list<OpNode*> starts;
  {
    const auto& ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](OpNode* in_node) {
        if (backbone_op_nodes.count(in_node) > 0) { Handler(in_node); }
      });
    };
    const auto& IsSource = [&](OpNode* node) {
      size_t in_size = 0;
      ForEachIn(node, [&](OpNode*) { ++in_size; });
      return in_size == 0;
    };
    for (OpNode* op_node : backbone_op_nodes) {
      if (IsSource(op_node)) { starts.push_back(op_node); }
    }
  }
  const auto& ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (backbone_op_nodes.count(out_node) > 0) { Handler(out_node); }
    });
  };
  op_graph.BfsForEachNode(starts, ForEachOut, Handler);
  return Maybe<void>::Ok();
}

Maybe<void> MakeGetterSspParallelConf4Depth(
    const std::list<std::unique_ptr<std::vector<OpNode*>>>& sequantial_trainable_fw_ops,
    std::function<Maybe<const ParallelConf&>(int64_t)>* SspParallelConf4Depth) {
  TODO();
  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow
