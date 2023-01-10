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
#include <glog/logging.h>
#include <cstdint>
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

int64_t GetSoleOutBlobSize(const OpNode* node) {
  const BlobDesc& blob_desc =
      node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi(node->op().SoleObn()));
  return blob_desc.shape().elem_cnt() * GetSizeOfDataType(blob_desc.data_type());
}

class DataParallelNodeSequence final {
 public:
  DataParallelNodeSequence(std::vector<const OpNode*> nodes, int64_t order)
      : nodes_(std::move(nodes)), order_(order), len_(nodes_.size()) {
    const OpNode* var_node = nodes_.front();
    CHECK(var_node->op().op_conf().has_variable_conf());
    model_size_ = GetSoleOutBlobSize(var_node);
  }
  ~DataParallelNodeSequence() = default;

  const OpNode* GetVariableNode() const { return nodes_.front(); }

  const OpNode* GetLastNode() const { return nodes_.back(); }

  int64_t order() const { return order_; }

  const std::vector<const OpNode*>& nodes() const { return nodes_; }

  const ParallelDesc& parallel_desc() const { return nodes_.front()->parallel_desc(); }

  int64_t model_size() const { return model_size_; }

  int64_t len() const { return len_; }

  void resize(const int64_t size) {
    CHECK_LE(size, len_);
    CHECK_GE(size, 1);
    nodes_.resize(size);
    len_ = nodes().size();
  }

 private:
  std::vector<const OpNode*> nodes_;
  int64_t order_;
  int64_t model_size_;
  int64_t len_;
};

using SequencePtr = std::shared_ptr<DataParallelNodeSequence>;

ParallelConf NonDistributedParallelConf4ParallelId(const ParallelDesc& pd,
                                                   const int64_t parallel_id) {
  std::string device_name;
  device_name += std::to_string(CHECK_JUST(pd.MachineId4ParallelId(parallel_id)));
  device_name += ":";
  device_name += std::to_string(CHECK_JUST(pd.DeviceId4ParallelId(parallel_id)));
  ParallelConf parallel_conf;
  *parallel_conf.mutable_device_name()->Add() = device_name;
  parallel_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(pd.device_type())));
  return parallel_conf;
}

Maybe<void> GetDataParallelVariableAndNaiveSuccNode(
    const OpNode* start, const std::function<bool(const OpNode*)>& IsAllowed,
    std::vector<const OpNode*>* out) {
  // Find sequence like: vairable -> cast_fp32_to_fp16
  if (!start->op().op_conf().has_variable_conf()) { return Maybe<void>::Ok(); }
  const ParallelDesc& pd = start->parallel_desc();
  if (pd.parallel_num() == 1) { return Maybe<void>::Ok(); }
  const OpNode* cur_node = start;
  while (cur_node != nullptr) {
    if (cur_node != start) {
      if (cur_node->parallel_desc() != pd) { break; }
      if (cur_node->in_edges().size() > 1) { break; }
      if (cur_node->op().input_bns().size() != 1) { break; }
      const std::string& sole_ibn = cur_node->op().SoleIbn();
      const NdSbp& ibn_nd_sbp = cur_node->NdSbp4BnInOp(sole_ibn);
      bool has_broadcast = false;
      FOR_RANGE(int, i, 0, ibn_nd_sbp.sbp_parallel_size()) {
        if (ibn_nd_sbp.sbp_parallel(i).has_broadcast_parallel()) { has_broadcast = true; };
      }
      if (!has_broadcast) { break; }
    }
    if (cur_node->op().output_bns().size() != 1) { break; }
    const std::string& sole_obn = cur_node->op().SoleObn();
    const NdSbp& obn_nd_sbp = cur_node->NdSbp4BnInOp(sole_obn);
    bool has_broadcast = false;
    FOR_RANGE(int, i, 0, obn_nd_sbp.sbp_parallel_size()) {
      if (obn_nd_sbp.sbp_parallel(i).has_broadcast_parallel()) { has_broadcast = true; };
    }
    if (!has_broadcast) { break; }
    out->emplace_back(cur_node);
    if (cur_node->out_edges().size() == 1) {
      cur_node = cur_node->SoleOutEdge()->dst_node();
    } else {
      cur_node = nullptr;
    }
  }
  return Maybe<void>::Ok();
}

void SetBroadcastParallel4OpNodeIbn(JobBuilder* builder, const OpNode* node,
                                    const std::string& ibn) {
  OpBlobArg op_blob_arg;
  op_blob_arg.set_op_name(node->op().op_name());
  op_blob_arg.set_bn_in_op(ibn);
  SbpParallel sbp_parallel;
  sbp_parallel.mutable_broadcast_parallel();
  builder->SetSbpParallel4Oba(op_blob_arg, sbp_parallel);
}

void SetBroadcastParallel4Consumers(JobBuilder* builder, const SequencePtr& sequence) {
  const OpNode* node = sequence->GetLastNode();
  const LogicalBlobId& lbi = node->op().BnInOp2Lbi(node->op().SoleObn());
  node->ForEachNodeOnOutEdge([&](const OpNode* out_node) {
    for (const std::string& ibn : out_node->op().input_bns()) {
      if (out_node->op().BnInOp2Lbi(ibn) == lbi) {
        SetBroadcastParallel4OpNodeIbn(builder, out_node, ibn);
      }
    }
  });
}

void SetNdSbp4OpNodeIbn(JobBuilder* builder, const OpNode* node, const std::string& ibn,
                        const NdSbp& nd_sbp) {
  OpBlobArg op_blob_arg;
  op_blob_arg.set_op_name(node->op().op_name());
  op_blob_arg.set_bn_in_op(ibn);
  builder->SetNdSbp4Oba(op_blob_arg, nd_sbp);
}

void SetNdSbp4Consumers(JobBuilder* builder, const SequencePtr& sequence, const NdSbp& nd_sbp) {
  const OpNode* node = sequence->GetLastNode();
  const LogicalBlobId& lbi = node->op().BnInOp2Lbi(node->op().SoleObn());
  const int64_t shard_restore_level =
      builder->job().job_conf().optimizer_placement_optimization_shard_restore_level();
  // If shard_restore_level == 0, no limit on consumer
  if (shard_restore_level == 1) {
    // Input lbn for parallel cast op
    std::string parallel_cast_input_lbn = GenLogicalBlobName(lbi);
    // Add parallel cast op to make soft limt on consumer to consume weight with Broadcast SBP.
    const auto parallel_cast_op =
        user_op::UserOpConfWrapperBuilder("System-ZeRO-ParallelCast-" + node->op().op_name() + "-"
                                          + NewUniqueId())
            .Op("hierarchical_parallel_cast")
            .Input("in", parallel_cast_input_lbn)
            .Output("out")
            .Attr<std::vector<std::string>>("nd_sbp", NdSbpToStringList(nd_sbp))
            .Attr<std::string>("grad_mode", "identity")  // don't do ndsbp cast at backward
            .Attr<std::vector<std::string>>("grad_nd_sbp", std::vector<std::string>())
            .ScopeSymbolId(node->op().op_conf().scope_symbol_id())
            .Build();
    builder->AddOps(node->parallel_desc().parallel_conf(), {parallel_cast_op.op_conf()});

    // Make consumers to consume parallel cast op
    auto out_lbn = parallel_cast_op.output("out", 0);
    node->ForEachNodeOnOutEdge([&](const OpNode* out_node) {
      for (const std::string& ibn : out_node->op().input_bns()) {
        if (out_node->op().BnInOp2Lbi(ibn) == lbi) {
          if (!CHECK_JUST(builder->IsInMutOpTransaction(out_node->op().op_name()))) {
            CHECK_JUST(builder->MutOpTransactionMut(out_node->op().op_conf()));
          }
          OperatorConf& mut_consumer_op =
              CHECK_JUST(builder->MutOpTransactionGet(out_node->op().op_name()));
          const auto& old_lbn = ReplaceInputLbnInOpCustomizedConf(&mut_consumer_op, ibn, out_lbn);
          CHECK_EQ(old_lbn, GenLogicalBlobName(lbi));
        }
      }
    });
  } else if (shard_restore_level == 2) {
    // Hard limt consumer to consume weight as Broadcast.
    node->ForEachNodeOnOutEdge([&](const OpNode* out_node) {
      for (const std::string& ibn : out_node->op().input_bns()) {
        if (out_node->op().BnInOp2Lbi(ibn) == lbi) {
          SetNdSbp4OpNodeIbn(builder, out_node, ibn, nd_sbp);
        }
      }
    });
  }
}

std::function<int64_t(const OpNode*)> MakeGetterOpNode2TopoOrder(const OpGraph& op_graph) {
  HashMap<const OpNode*, int64_t> op_node2topo_order;
  int64_t node_cnt = 0;
  op_graph.TopoForEachNode([&](const OpNode* node) {
    op_node2topo_order[node] = node_cnt;
    node_cnt += 1;
  });
  return [op_node2topo_order](const OpNode* node) { return op_node2topo_order.at(node); };
}

int64_t GetMinConsumerOrder(const OpGraph& op_graph, const OpNode* node,
                            const std::function<int64_t(const OpNode*)>& OpNode2Order) {
  int64_t min_consumer_topo_order = op_graph.node_num();
  node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
    min_consumer_topo_order = std::min(min_consumer_topo_order, OpNode2Order(dst));
  });
  return min_consumer_topo_order;
}

void ForEachDataParallelNodeSequence(const OpGraph& op_graph,
                                     const std::function<bool(const OpNode*)>& IsAllowed,
                                     std::function<void(SequencePtr&&)> Handler) {
  auto OpNode2Order = MakeGetterOpNode2TopoOrder(op_graph);
  op_graph.ForEachNode([&](const OpNode* node) {
    std::vector<const OpNode*> nodes;
    // Find sequence like: vairable -> cast_fp32_to_fp16
    CHECK_JUST(GetDataParallelVariableAndNaiveSuccNode(node, IsAllowed, &nodes));
    if (nodes.empty()) { return; }
    const int64_t order = GetMinConsumerOrder(op_graph, nodes.back(), OpNode2Order);
    Handler(std::make_shared<DataParallelNodeSequence>(std::move(nodes), order));
  });
}

bool SequenceCompSortedByOrderAsc(const SequencePtr& lhs, const SequencePtr& rhs) {
  return lhs->order() < rhs->order();
}

bool SequenceCompSortedByModelSizeDesc(const SequencePtr& lhs, const SequencePtr& rhs) {
  return lhs->model_size() > rhs->model_size();
}

void ForEachParallelSortedNodeSequence(
    const OpGraph& op_graph, const std::function<bool(const OpNode*)>& IsAllowed,
    const std::function<bool(const SequencePtr&, const SequencePtr&)>& Comp,
    const std::function<void(const ParallelDesc&, std::vector<SequencePtr>&&)>& Handler) {
  HashMap<ParallelDesc, std::vector<SequencePtr>> parallel_desc2sequences;
  // Find sequence like: vairable -> cast_fp32_to_fp16
  ForEachDataParallelNodeSequence(op_graph, IsAllowed, [&](SequencePtr&& sequence) {
    parallel_desc2sequences[sequence->parallel_desc()].emplace_back(std::move(sequence));
  });
  for (auto& pair : parallel_desc2sequences) {
    auto& sequences = pair.second;
    std::sort(sequences.begin(), sequences.end(), Comp);
    Handler(pair.first, std::move(sequences));
  }
}

bool IsS0Parallel(const SbpParallel& sbp_parallel) {
  return sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == 0;
}

bool IsS0Parallel(const SbpSignature& signature, const std::string& bn) {
  return IsS0Parallel(signature.bn_in_op2sbp_parallel().at(bn));
}

bool IsNdSbpMatch(const NdSbpSignature& signature, const std::string& bn, const NdSbp& nd_sbp) {
  return signature.bn_in_op2nd_sbp().at(bn) == nd_sbp;
}

bool IsNdSbpSupported4Op(const OpNode* node, const NdSbp& nd_sbp) {
  if (node->op().input_bns().size() != 1 || node->op().output_bns().size() != 1) { return false; }
  std::vector<NdSbpSignature> list;
  auto LogicalBlobDesc4Ibn = [&](const std::string& bn) -> Maybe<const BlobDesc&> {
    return Maybe<const BlobDesc&>(node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi(bn)));
  };
  CHECK_JUST(node->op().GetNdSbpSignatureList(LogicalBlobDesc4Ibn, node->parallel_desc(), &list));
  const auto IsInAndOutMatch = [&](const NdSbpSignature& signature) {
    return IsNdSbpMatch(signature, node->op().SoleIbn(), nd_sbp)
           && IsNdSbpMatch(signature, node->op().SoleObn(), nd_sbp);
  };
  return std::any_of(list.cbegin(), list.cend(), IsInAndOutMatch);
}

bool IsS0SignatureSupported(const OpNode* node) {
  if (node->op().input_bns().size() != 1 || node->op().output_bns().size() != 1) { return false; }
  SbpSignatureList list;
  auto LogicalBlobDesc4Ibn = [&](const std::string& bn) -> Maybe<const BlobDesc&> {
    return Maybe<const BlobDesc&>(node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi(bn)));
  };
  CHECK_JUST(node->op().GetSbpSignaturesIf(LogicalBlobDesc4Ibn,
                                           node->parallel_desc().parallel_num(), &list));
  const auto IsInOutS0Parallel = [&](const SbpSignature& signature) {
    return IsS0Parallel(signature, node->op().SoleIbn())
           && IsS0Parallel(signature, node->op().SoleObn());
  };
  return std::any_of(list.sbp_signature().cbegin(), list.sbp_signature().cend(), IsInOutS0Parallel);
}

void ForEachModelSizeBalancedPartition(
    const ParallelDesc& parallel_desc, std::vector<SequencePtr>&& sorted_sequences,
    const std::function<void(ParallelDesc new_parallel_desc, std::vector<SequencePtr>&&)>&
        Handler) {
  std::vector<SequencePtr> sequences = std::move(sorted_sequences);
  std::vector<int64_t> parallel_id2model_size(parallel_desc.parallel_num(), 0);
  std::vector<std::vector<SequencePtr>> partitions(parallel_desc.parallel_num());
  for (auto& sequence : sequences) {
    const auto it =
        std::min_element(parallel_id2model_size.cbegin(), parallel_id2model_size.cend());
    const int64_t min_parallel_id = std::distance(parallel_id2model_size.cbegin(), it);
    parallel_id2model_size.at(min_parallel_id) += sequence->model_size();
    partitions.at(min_parallel_id).emplace_back(std::move(sequence));
  }
  for (int64_t i = 0; i < parallel_desc.parallel_num(); ++i) {
    ParallelConf parallel_conf = NonDistributedParallelConf4ParallelId(parallel_desc, i);
    Handler(parallel_conf, std::move(partitions.at(i)));
  }
}

namespace {
bool IsSplitValid(const Shape& shape, const NdSbp& nd_sbp, const Shape& hierachy,
                  int64_t min_size) {
  if (shape.NumAxes() < 1 || shape.elem_cnt() < 1) { return false; }
  CHECK_EQ(nd_sbp.sbp_parallel_size(), hierachy.NumAxes());
  Shape cur_shape = shape;
  if (cur_shape.elem_cnt() < min_size) { return false; }
  FOR_RANGE(int64_t, i, 0, hierachy.NumAxes()) {
    const auto& sbp = nd_sbp.sbp_parallel(i);
    if (sbp.has_split_parallel()) {
      const int64_t dim = sbp.split_parallel().axis();
      if (dim >= cur_shape.NumAxes()) { return false; }
      // Unbalanced split and take the minimum
      cur_shape.Set(dim, cur_shape.At(dim) / hierachy.At(i));
      // Larger then min size.
      if (cur_shape.elem_cnt() < min_size) { return false; }
    }
  }
  return true;
}

void GenerateSplitSignature(const NdSbp& var_nd_sbp, const OperatorConf& new_var_op_conf,
                            std::string& new_split_signature, int64_t& split_dim) {
  if (new_var_op_conf.variable_conf().nd_sbp_size() > 0 && NdSbpIsAllBroadcast(var_nd_sbp)) {
    // split last dim
    split_dim = new_var_op_conf.variable_conf().nd_sbp_size() - 1;
    // All B, B -> S0
    new_split_signature = "S(0)";
  } else {
    // ND sbp, (*, B, S, *) -> (*, S, S, *)
    // ND sbp, (*, S, B, *) -> (*, S, S, *)
    FOR_RANGE(int64_t, j, 0, new_var_op_conf.variable_conf().nd_sbp_size()) {
      if (new_var_op_conf.variable_conf().nd_sbp(j) == "B") {
        std::vector<int64_t> adjacent_dim{j - 1, j + 1};
        for (auto const& dim_to_try : adjacent_dim) {
          if (dim_to_try >= 0 && dim_to_try < new_var_op_conf.variable_conf().nd_sbp_size()) {
            SbpParallel sbp;
            if (ParseSbpParallelFromString(new_var_op_conf.variable_conf().nd_sbp(dim_to_try), &sbp)
                && sbp.has_split_parallel()) {
              new_split_signature = new_var_op_conf.variable_conf().nd_sbp(dim_to_try);
              split_dim = j;
            }
          }
          if (new_split_signature != "") break;
        }
      }
      // Only split one more dim.
      if (new_split_signature != "") break;
    }
  }
}
void ShardSequence(JobBuilder* builder, const int64_t threshold, const ParallelDesc& pd,
                   std::vector<SequencePtr>&& sorted_sequences) {
  // For all sorted sequence, set the variable op in the sequence to S
  // and add ctrl edge to control the execution order between variable ops.
  // A sequence is a variable op and its cast(fp32 to fp16) op. This is because the forward pass
  // consume the fp16 variable and the optimizer consume the fp32 variable.
  std::string prev_allowed_op_name = "";
  for (int64_t i = 0; i < sorted_sequences.size(); ++i) {
    const OpNode* var_node = sorted_sequences.at(i)->GetVariableNode();
    OperatorConf new_var_op_conf = var_node->op().op_conf();
    const std::string& sole_obn = var_node->op().SoleObn();
    const NdSbp& var_nd_sbp = var_node->NdSbp4BnInOp(sole_obn);
    const Shape& logical_shape = Shape(new_var_op_conf.variable_conf().shape());

    std::string new_split_signature = "";
    int64_t split_dim = 0;
    GenerateSplitSignature(var_nd_sbp, new_var_op_conf, new_split_signature, split_dim);
    if (new_split_signature != "") {
      *new_var_op_conf.mutable_variable_conf()->mutable_nd_sbp(split_dim) = new_split_signature;
    } else {
      continue;
    }

    bool split_is_allowed = true;
    {
      NdSbp new_nd_sbp;
      std::vector<std::string> nd_sbp_str_vec;
      for (const auto& sbp_str : new_var_op_conf.variable_conf().nd_sbp()) {
        nd_sbp_str_vec.emplace_back(sbp_str);
      }
      ParseNdSbpFromStringList(nd_sbp_str_vec, &new_nd_sbp);
      // check allowed by min shard size and evenly split
      if (split_is_allowed) {
        split_is_allowed = IsSplitValid(logical_shape, new_nd_sbp, *pd.hierarchy(), threshold);
      }
      if (split_is_allowed) {
        // resize sequence by new nd sbp limit
        auto& cur_seq = sorted_sequences.at(i);
        int64_t max_len = 1;
        if (cur_seq->len() > 1) {
          FOR_RANGE(int64_t, node_idx, 1, cur_seq->len()) {
            if (IsNdSbpSupported4Op(cur_seq->nodes().at(node_idx), new_nd_sbp)) {
              ++max_len;
            } else {
              break;
            }
          }
        }
        if (max_len < cur_seq->len()) { cur_seq->resize(max_len); }
      }
    }
    if (!split_is_allowed) {
      VLOG(3) << var_node->op().op_name() << " failed to change from B to S "
              << " with op conf " << new_var_op_conf.variable_conf().DebugString();
      continue;
    }
    if (!prev_allowed_op_name.empty()) {
      new_var_op_conf.add_ctrl_in_op_name(prev_allowed_op_name);
    }
    builder->MutOpsOnlyOnce({new_var_op_conf});
    // Set consumers to consum this variable op's cast op's output as Broadcast.
    if (new_split_signature != "") {
      SetNdSbp4Consumers(builder, sorted_sequences.at(i), var_nd_sbp);
    }
    prev_allowed_op_name = var_node->op().op_name();
    VLOG(3) << var_node->op().op_name() << " succeed to change from B to " << new_split_signature
            << " on ranks dim " << split_dim << " with op conf "
            << new_var_op_conf.variable_conf().DebugString();
  }
}
}  // namespace

Maybe<void> RewriteDistributedSplit(const OpGraph& op_graph, JobBuilder* builder) {
  const int64_t threshold = builder->job().job_conf().optimizer_placement_optimization_threshold();
  const auto IsAllowed = [](const OpNode* n) -> bool {
    // No need to limit here.
    return true;
  };
  const auto PlacementSequencesAsSplitParallel = [&](const ParallelDesc& pd,
                                                     std::vector<SequencePtr>&& sorted_sequences) {
    ShardSequence(builder, threshold, pd, std::forward<std::vector<SequencePtr>>(sorted_sequences));
  };
  ForEachParallelSortedNodeSequence(op_graph, IsAllowed, SequenceCompSortedByOrderAsc,
                                    PlacementSequencesAsSplitParallel);
  JUST(builder->MutOpTransactionCommit());
  return Maybe<void>::Ok();
}

Maybe<void> RewriteNonDistributed(const OpGraph& op_graph, JobBuilder* builder) {
  HashMap<ParallelDesc, std::vector<SequencePtr>> new_parallel_desc2sequences;
  const auto RewritePartition = [&](const ParallelDesc& new_parallel_desc,
                                    std::vector<SequencePtr>&& partition) {
    for (auto& sequence : partition) {
      for (const OpNode* op_node : sequence->nodes()) {
        builder->MutParallelConfOnlyOnce(op_node->op().op_name(),
                                         new_parallel_desc.parallel_conf());
      }
      SetBroadcastParallel4Consumers(builder, sequence);
      new_parallel_desc2sequences[new_parallel_desc].emplace_back(std::move(sequence));
    }
  };
  const auto RewriteSequences = [&](const ParallelDesc& pd,
                                    std::vector<SequencePtr>&& sorted_sequences) {
    ForEachModelSizeBalancedPartition(pd, std::move(sorted_sequences), RewritePartition);
  };
  const int64_t threshold = builder->job().job_conf().optimizer_placement_optimization_threshold();
  const auto IsAllowed = [threshold](const OpNode* n) -> bool {
    if (n->op().op_conf().has_variable_conf()) {
      const Shape shape(n->op().op_conf().variable_conf().shape());
      const int64_t parallel_num = n->parallel_desc().parallel_num();
      return shape.elem_cnt() >= threshold * parallel_num;
    } else {
      return true;
    }
  };
  ForEachParallelSortedNodeSequence(op_graph, IsAllowed, SequenceCompSortedByModelSizeDesc,
                                    RewriteSequences);

  for (auto& parallel_desc7sequences : new_parallel_desc2sequences) {
    auto& sequences = parallel_desc7sequences.second;
    std::sort(sequences.begin(), sequences.end(), SequenceCompSortedByOrderAsc);
    for (int64_t i = 1; i < sequences.size(); ++i) {
      const OpNode* cur_var_node = sequences.at(i)->GetVariableNode();
      OperatorConf cur_var_conf(cur_var_node->op().op_conf());
      const OpNode* prev_var_node = sequences.at(i - i)->GetVariableNode();
      cur_var_conf.add_ctrl_in_op_name(prev_var_node->op().op_name());
      builder->MutOpsOnlyOnce({cur_var_conf});
    }
  }
  return Maybe<void>::Ok();
}

class OptimizerPlacementOptimizationPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OptimizerPlacementOptimizationPass);
  OptimizerPlacementOptimizationPass() = default;
  ~OptimizerPlacementOptimizationPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!(ctx->job_desc().IsTrain()
          && ctx->job_desc().job_conf().has_optimizer_placement_optimization_mode()
          && ctx->job_desc().job_conf().optimizer_placement_optimization_mode() != "none")) {
      return Maybe<void>::Ok();
    }
    if (job->job_conf().enable_auto_parallel()
        && job->job_conf().enable_auto_parallel_ignore_user_sbp_config()) {
      LOG(WARNING) << "ZeRO optimization will be ignored when enabling AutoParallel to ignore user "
                      "sbp configuration";
      if (job->job_conf().enable_auto_memory() != oneflow::AutoMemoryStrategy::kHeavyAutoMemory) {
        job->mutable_job_conf()->set_enable_auto_memory(
            ::oneflow::AutoMemoryStrategy::kModerateAutoMemory);
        LOG(WARNING) << "But we turn on moderate auto memory to reduce the memory, which has "
                        "similar effect as the ZeRO optimization";
      }
      return Maybe<void>::Ok();
    }
    const std::string& mode = ctx->job_desc().job_conf().optimizer_placement_optimization_mode();
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    if (mode == "non_distributed") {
      return RewriteNonDistributed(op_graph, &job_builder);
    } else if (mode == "distributed_split") {
      return RewriteDistributedSplit(op_graph, &job_builder);
    } else {
      return Error::UnimplementedError();
    }
  }
};

REGISTER_JOB_PASS("OptimizerPlacementOptimizationPass", OptimizerPlacementOptimizationPass);

}  // namespace

}  // namespace oneflow
