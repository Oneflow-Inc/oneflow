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
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/framework/interpreter.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace oneflow {

namespace {

class StagePartitionStragety {
 public:
  StagePartitionStragety() = default;
  ~StagePartitionStragety() = default;
  virtual Maybe<void> Apply(Job* job, JobPassCtx* ctx) const = 0;
};

class StagePartitionPass final : public JobPass {
 public:
  StagePartitionPass() = default;
  ~StagePartitionPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const std::string& partition_strategy = ctx->job_desc().String("stage_partition_strategy");
    std::unique_ptr<const StagePartitionStragety> strategy;
    strategy.reset(NewObj<std::string, StagePartitionStragety>(partition_strategy));
    return strategy->Apply(job, ctx);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain() && ctx.job_desc().Bool("enable_stage_partition");
  }
};

REGISTER_JOB_PASS("StagePartition", StagePartitionPass);

#define REGISTER_SSP_PARTITION_STRATEGY(strategy_name, strategy_type)        \
  REGISTER_CLASS_CREATOR(std::string, strategy_name, StagePartitionStragety, \
                         ([] { return new strategy_type(); }));

class DisableStagePartitionStrategy : public StagePartitionStragety {
 public:
  DisableStagePartitionStrategy() = default;
  ~DisableStagePartitionStrategy() = default;

  Maybe<void> Apply(Job* job, JobPassCtx*) const override { return Maybe<void>::Ok(); }
};
REGISTER_SSP_PARTITION_STRATEGY("disable", DisableStagePartitionStrategy);

class NaiveSequantialStagePartitionStrategy : public StagePartitionStragety {
 public:
  NaiveSequantialStagePartitionStrategy() = default;
  ~NaiveSequantialStagePartitionStrategy() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    auto op_graph = JUST(OpGraph::New(*job));
    JobBuilder job_builder(job);
    std::function<Maybe<int64_t>(int64_t old_scope, int64_t stage_scope)> GetMergedScopeSymbolId;
    MakeGetterGetMergedScopeSymbolId(&GetMergedScopeSymbolId);
    const auto& scope_storage = *Global<vm::SymbolStorage<Scope>>::Get();
    JUST(ForEachStageScope4TrainableFwOp(
        *op_graph, ctx->job_desc(),
        [&](const OpNode* op_node, int64_t stage_scope_symbol_id) -> Maybe<void> {
          const auto& old_op_conf = op_node->op().op_conf();
          CHECK_OR_RETURN(old_op_conf.has_scope_symbol_id());
          const auto& old_scope = JUST(scope_storage.MaybeGet(old_op_conf.scope_symbol_id()));
          int64_t merged_scope_symbol_id =
              JUST(GetMergedScopeSymbolId(old_op_conf.scope_symbol_id(), stage_scope_symbol_id));
          const auto& merged_scope = JUST(scope_storage.MaybeGet(merged_scope_symbol_id));
          // Sets scope_symbol_id
          std::vector<OperatorConf> op_confs(1);
          auto* new_op_conf = &op_confs.at(0);
          new_op_conf->CopyFrom(old_op_conf);
          new_op_conf->set_scope_symbol_id(merged_scope_symbol_id);
          job_builder.MutOpsOnlyOnce(op_confs);
          // Sets parallel_conf
          const auto& new_parallel_desc = JUST(merged_scope.GetParallelDesc(*new_op_conf));
          const auto& op_name = op_node->op().op_name();
          job_builder.MutParallelConfOnlyOnce(op_name, new_parallel_desc.parallel_conf());
          const auto& old_parallel_conf = JUST(old_scope.GetParallelDesc(old_op_conf));
          if (new_parallel_desc != old_parallel_conf) {
            LOG(INFO) << "======== " << old_op_conf.name() << " ========" << std::endl
                      << "-------- old --------" << std::endl
                      << old_parallel_conf.parallel_conf().DebugString() << "-------- new --------"
                      << std::endl
                      << new_parallel_desc.parallel_conf().DebugString();
          }
          return Maybe<void>::Ok();
        }));
    return Maybe<void>::Ok();
  }

 private:
  void MakeGetterGetMergedScopeSymbolId(
      std::function<Maybe<int64_t>(int64_t old_scope, int64_t stage_scope)>* GetMergedScopeSymbolId)
      const {
    using CacheT = HashMap<std::pair<int64_t, int64_t>, int64_t>;
    auto old7stage2merged = std::make_shared<CacheT>();
    *GetMergedScopeSymbolId = [old7stage2merged, this](int64_t old_scope_id,
                                                       int64_t stage_scope_id) -> Maybe<int64_t> {
      std::pair<int64_t, int64_t> old7stage(old_scope_id, stage_scope_id);
      const auto& iter = old7stage2merged->find(old7stage);
      if (iter != old7stage2merged->end()) { return iter->second; }
      int64_t merge_scope_symbol_id = JUST(MergeScope(old_scope_id, stage_scope_id));
      old7stage2merged->emplace(old7stage, merge_scope_symbol_id);
      return merge_scope_symbol_id;
    };
  }

  // Returns scope_symbol_id
  Maybe<int64_t> MergeScope(int64_t old_scope_id, int64_t stage_scope_id) const {
    const auto& storage = *Global<vm::SymbolStorage<Scope>>::Get();
    const auto& old_scope = JUST(storage.MaybeGet(old_scope_id));
    const auto& stage_scope = JUST(storage.MaybeGet(stage_scope_id));
    cfg::ScopeProto merged_scope;
    merged_scope.InitFromProto(old_scope.scope_proto());
    merged_scope.set_parent_scope_symbol_id(old_scope_id);
    merged_scope.set_device_parallel_desc_symbol_id(
        stage_scope.scope_proto().device_parallel_desc_symbol_id());
    merged_scope.set_host_parallel_desc_symbol_id(
        stage_scope.scope_proto().host_parallel_desc_symbol_id());
    auto* map = merged_scope.mutable_attr_name2attr_value();
    (*map)["stage_placement_id"].set_at_int64(stage_scope.Int64("stage_placement_id"));
    (*map)["stage_weight_buffer_size"].set_at_int64(stage_scope.Int64("stage_weight_buffer_size"));
    int64_t symbol_id = 0;
    JUST(LogicalInterpreter().Run([&](InstructionsBuilder* builder) -> Maybe<void> {
      symbol_id = JUST(builder->FindOrCreateSymbolId<cfg::ScopeProto>(merged_scope));
      return Maybe<void>::Ok();
    }));
    // TODO(lixinqi): Remove this urgly code after most python code migrated into cpp code
    {
      ScopeProto scope_proto;
      merged_scope.ToProto(&scope_proto);
      Global<ForeignCallback>::Get()->AddScopeToPyStorage(symbol_id, scope_proto.DebugString());
    }
    return symbol_id;
  }

  Maybe<void> ForEachStageScope4TrainableFwOp(
      const OpGraph& op_graph, const JobDesc& job_desc,
      const std::function<Maybe<void>(const OpNode*, int64_t scope_symbol_id)>& Handler) const {
    // Sequantialize trainable forward ops
    std::list<std::unique_ptr<std::vector<OpNode*>>> sequantialized_fw_ops;
    JUST(GetSequantialFwOps(op_graph, &sequantialized_fw_ops));
    // Gets stage partition config
    std::vector<int64_t> stage_partition_scope_ids;
    JUST(GetStagePartitionScopeIds(job_desc, &stage_partition_scope_ids));
    // Partition to stages
    std::function<Maybe<int64_t>(int64_t)> Stage4Depth;
    JUST(GetStageDepth2Stage(sequantialized_fw_ops, stage_partition_scope_ids, &Stage4Depth));
    int64_t depth = 0;
    for (const auto& fused_vec : sequantialized_fw_ops) {
      int64_t stage = JUST(Stage4Depth(depth));
      int64_t scope_symbol_id = JUST(VectorAt(stage_partition_scope_ids, stage));
      for (OpNode* op_node : *fused_vec) { JUST(Handler(op_node, scope_symbol_id)); }
      ++depth;
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSequantialFwOps(
      const OpGraph& op_graph,
      std::list<std::unique_ptr<std::vector<OpNode*>>>* sequantialized_fw_ops) const {
    HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>> backbone_op2fused_ops;
    JUST(GetBackboneOp2FusedOps(op_graph, &backbone_op2fused_ops));
    const auto& ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](OpNode* in_node) {
        if (backbone_op2fused_ops.count(in_node) > 0) { Handler(in_node); }
      });
    };
    std::list<OpNode*> starts;
    {
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
    JUST(op_graph.MaybeTopoForEachNode(
        starts, ForEachIn, ForEachOut, [&](OpNode* op_node) -> Maybe<void> {
          const auto& iter = backbone_op2fused_ops.find(op_node);
          CHECK_OR_RETURN(iter != backbone_op2fused_ops.end());
          sequantialized_fw_ops->emplace_back(std::move(iter->second));
          return Maybe<void>::Ok();
        }));
    return Maybe<void>::Ok();
  }

  Maybe<void> GetStageDepth2Stage(
      const std::list<std::unique_ptr<std::vector<OpNode*>>>& sequantialized_fw_ops,
      const std::vector<int64_t>& stage_partition_scope_ids,
      std::function<Maybe<int64_t>(int64_t)>* Stage4Depth) const {
    CHECK_LE_OR_RETURN(stage_partition_scope_ids.size(), sequantialized_fw_ops.size())
        << "Partition failed. Number of stages is bigger than number of ops";
    const auto& scope_storage = *Global<vm::SymbolStorage<Scope>>::Get();
    int64_t num_depth = sequantialized_fw_ops.size();
    CHECK_GT_OR_RETURN(num_depth, 0);
    int64_t num_stages = stage_partition_scope_ids.size();
    CHECK_GT_OR_RETURN(num_stages, 0);
    std::vector<float> stage_payloads(num_stages);
    float total_payloads = 0;
    for (int64_t i = 0; i < num_stages; ++i) {
      const auto& scope = JUST(scope_storage.MaybeGet(stage_partition_scope_ids.at(i)));
      int64_t stage_payload = scope.Int64("stage_load");
      CHECK_GT_OR_RETURN(stage_payload, 0);
      stage_payloads.at(i) = stage_payload;
      total_payloads += stage_payload;
    }
    using RangeListT = std::vector<std::pair<int64_t, int64_t>>;
    auto stage2depth_ranges = std::make_shared<RangeListT>(num_stages);
    float depth_from_start = 0;
    for (int64_t i = 0; i < num_stages; ++i) {
      stage2depth_ranges->at(i).first = static_cast<int64_t>(depth_from_start);
      depth_from_start += num_depth * stage_payloads.at(i) / total_payloads;
      stage2depth_ranges->at(i).second = static_cast<int64_t>(depth_from_start);
    }
    stage2depth_ranges->at(0).first = 0;
    stage2depth_ranges->at(num_stages - 1).second = num_depth;
    *Stage4Depth = [stage2depth_ranges](int64_t depth) -> Maybe<int64_t> {
      for (int i = 0; i < stage2depth_ranges->size(); ++i) {
        const auto& range = stage2depth_ranges->at(i);
        if (depth >= range.first && depth < range.second) { return i; }
      }
      OF_UNIMPLEMENTED() << "depth: " << depth;
    };
    return Maybe<void>::Ok();
  }

  Maybe<void> GetTrainableFwOps(const OpGraph& op_graph, HashSet<OpNode*>* trainable_fw_ops) const {
    std::function<bool(OpNode*)> NeedBackwardOp;
    JUST(MakePredicatorNeedBackwardOp(op_graph, &NeedBackwardOp));
    op_graph.ForEachNode([&](OpNode* node) {
      if (NeedBackwardOp(node)) { trainable_fw_ops->insert(node); }
    });
    return Maybe<void>::Ok();
  }

  Maybe<void> GetBackboneOp2FusedOps(
      const OpGraph& op_graph,
      HashMap<OpNode*, std::unique_ptr<std::vector<OpNode*>>>* backbone_op2fused_ops) const {
    // Gets trainable forward ops.
    HashSet<OpNode*> trainable_fw_ops;
    JUST(GetTrainableFwOps(op_graph, &trainable_fw_ops));
    // Gets backbone ops.
    HashSet<OpNode*> backbone_op_nodes;
    JUST(GetBackBoneOps(op_graph, trainable_fw_ops, &backbone_op_nodes));
    // Fuses forward ops to backbone ops.
    HashMap<OpNode*, OpNode*> fw_op2backbone_op;
    JUST(FuseFwOpsToBackboneOps(op_graph, backbone_op_nodes, &fw_op2backbone_op));
    for (const auto& pair : fw_op2backbone_op) {
      auto* fused_ops = &(*backbone_op2fused_ops)[pair.second];
      if (!*fused_ops) { fused_ops->reset(new std::vector<OpNode*>()); }
      (*fused_ops)->push_back(pair.first);
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
                             HashSet<OpNode*>* backbone_op_nodes) const {
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

  Maybe<void> FuseFwOpsToBackboneOps(const OpGraph& op_graph,
                                     const HashSet<OpNode*>& backbone_op_nodes,
                                     HashMap<OpNode*, OpNode*>* fw_op2backbone_op) const {
    using namespace std::placeholders;
    // Fuses nearest inputs
    const auto& ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& DoEach) {
      node->ForEachNodeOnInEdge([&](OpNode* in_node) {
        if (fw_op2backbone_op->count(in_node) == 0) { DoEach(in_node); }
      });
    };
    JUST(TopoForEachBackboneOp(op_graph, backbone_op_nodes, [&](OpNode* backbone_op_node) {
      op_graph.BfsForEachNode({backbone_op_node}, ForEachIn, [&](OpNode* node) {
        OpNode** backbone_op_node_ptr = &(*fw_op2backbone_op)[node];
        if (*backbone_op_node_ptr == nullptr) { *backbone_op_node_ptr = backbone_op_node; }
      });
    }));
    // Fuses nearest outputs
    const auto& ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& DoEach) {
      node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
        if (fw_op2backbone_op->count(out_node) == 0) { DoEach(out_node); }
      });
    };
    JUST(ReverseTopoForEachBackboneOp(op_graph, backbone_op_nodes, [&](OpNode* backbone_op_node) {
      op_graph.BfsForEachNode({backbone_op_node}, ForEachOut, [&](OpNode* node) {
        OpNode** backbone_op_node_ptr = &(*fw_op2backbone_op)[node];
        if (*backbone_op_node_ptr == nullptr) { *backbone_op_node_ptr = backbone_op_node; }
      });
    }));
    // Fuses nearest remainder inputs and outputs
    HashSet<const OpNode*> visisted;
    const auto& ForEachNext = [&](OpNode* node, const std::function<void(OpNode*)>& DoEach) {
      node->ForEachNodeOnInOutEdge([&](OpNode* next_node) {
        if (visisted.count(next_node) == 0) { DoEach(next_node); }
      });
    };
    JUST(ReverseTopoForEachBackboneOp(op_graph, backbone_op_nodes, [&](OpNode* backbone_op_node) {
      op_graph.BfsForEachNode({backbone_op_node}, ForEachNext, [&](OpNode* node) {
        OpNode** backbone_op_node_ptr = &(*fw_op2backbone_op)[node];
        if (*backbone_op_node_ptr == nullptr) { *backbone_op_node_ptr = backbone_op_node; }
        visisted.insert(node);
      });
    }));
    return Maybe<void>::Ok();
  }

  Maybe<void> ReverseTopoForEachBackboneOp(const OpGraph& op_graph,
                                           const HashSet<OpNode*>& backbone_op_nodes,
                                           const std::function<void(OpNode*)>& Handler) const {
    const auto& ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
        if (backbone_op_nodes.count(out_node) > 0) { Handler(out_node); }
      });
    };
    std::list<OpNode*> starts;
    {
      const auto& IsSink = [&](OpNode* node) {
        size_t num_outputs = 0;
        ForEachOut(node, [&](OpNode*) { ++num_outputs; });
        return num_outputs == 0;
      };
      for (OpNode* op_node : backbone_op_nodes) {
        if (IsSink(op_node)) { starts.push_back(op_node); }
      }
    }
    const auto& ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](OpNode* in_node) {
        if (backbone_op_nodes.count(in_node) > 0) { Handler(in_node); }
      });
    };
    op_graph.TopoForEachNode(starts, ForEachOut, ForEachIn, Handler);
    return Maybe<void>::Ok();
  }

  Maybe<void> TopoForEachBackboneOp(const OpGraph& op_graph,
                                    const HashSet<OpNode*>& backbone_op_nodes,
                                    const std::function<void(OpNode*)>& Handler) const {
    const auto& ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](OpNode* in_node) {
        if (backbone_op_nodes.count(in_node) > 0) { Handler(in_node); }
      });
    };
    std::list<OpNode*> starts;
    {
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
    op_graph.TopoForEachNode(starts, ForEachIn, ForEachOut, Handler);
    return Maybe<void>::Ok();
  }

  Maybe<void> GetStagePartitionScopeIds(const JobDesc& job_desc,
                                        std::vector<int64_t>* stage_partition_scope_ids) const {
    const auto& scope_ids = job_desc.ListInt64("stage_partition_scope_ids");
    CHECK_GT_OR_RETURN(scope_ids.size(), 0);
    stage_partition_scope_ids->assign(scope_ids.begin(), scope_ids.end());
    return Maybe<void>::Ok();
  }
};
REGISTER_SSP_PARTITION_STRATEGY("naive_sequantial", NaiveSequantialStagePartitionStrategy);

}  // namespace

}  // namespace oneflow
