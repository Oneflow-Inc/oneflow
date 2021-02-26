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
#include "oneflow/core/common/util.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"

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
      : nodes_(std::move(nodes)), order_(order) {
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

 private:
  std::vector<const OpNode*> nodes_;
  int64_t order_;
  int64_t model_size_;
};

using SequencePtr = std::shared_ptr<const DataParallelNodeSequence>;

ParallelConf NonDistributedParallelConf4ParallelId(const ParallelDesc& pd,
                                                   const int64_t parallel_id) {
  std::string device_name;
  device_name += std::to_string(CHECK_JUST(pd.MachineId4ParallelId(parallel_id)));
  device_name += ":";
  device_name += std::to_string(CHECK_JUST(pd.DeviceId4ParallelId(parallel_id)));
  ParallelConf parallel_conf;
  *parallel_conf.mutable_device_name()->Add() = device_name;
  parallel_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(pd.device_type())));
  return parallel_conf;
}

Maybe<void> GetDataParallelVariableAndNaiveSuccNode(
    const OpNode* start, const std::function<bool(const OpNode*)>& IsAllowed,
    std::vector<const OpNode*>* out) {
  if (!start->op().op_conf().has_variable_conf()) { return Maybe<void>::Ok(); }
  const ParallelDesc& pd = start->parallel_desc();
  if (pd.device_type() != DeviceType::kGPU) { return Maybe<void>::Ok(); }
  if (pd.parallel_num() == 1) { return Maybe<void>::Ok(); }
  const OpNode* cur_node = start;
  while (cur_node != nullptr) {
    if (cur_node != start) {
      if (cur_node->parallel_desc() != pd) { break; }
      if (cur_node->in_edges().size() > 1) { break; }
      if (cur_node->op().input_bns().size() != 1) { break; }
      const std::string& sole_ibn = cur_node->op().SoleIbn();
      if (!cur_node->SbpParallel4BnInOp(sole_ibn).has_broadcast_parallel()) { break; }
    }
    if (!IsAllowed(cur_node)) { break; }
    if (cur_node->op().output_bns().size() != 1) { break; }
    const std::string& sole_obn = cur_node->op().SoleObn();
    if (!cur_node->SbpParallel4BnInOp(sole_obn).has_broadcast_parallel()) { break; }
    out->push_back(cur_node);
    if (cur_node->out_edges().size() == 1) {
      cur_node = cur_node->SoleOutEdge()->dst_node();
    } else {
      cur_node = nullptr;
    }
  }
  return Maybe<void>::Ok();
}

void ForEachOutNodeConsumingLbi(
    const OpNode* node, const LogicalBlobId& lbi,
    const std::function<void(const OpNode* out_node, const std::string& ibn)>& Handler) {
  node->ForEachNodeOnOutEdge([&](const OpNode* out_node) {
    for (const std::string& ibn : out_node->op().input_bns()) {
      if (out_node->op().BnInOp2Lbi(ibn) == lbi) { Handler(out_node, ibn); }
    }
  });
}

void ForEachOutNodeConsumingSoleOut(
    const OpNode* node,
    const std::function<void(const OpNode* out_node, const std::string& ibn)>& Handler) {
  ForEachOutNodeConsumingLbi(node, node->op().BnInOp2Lbi(node->op().SoleObn()), Handler);
}

void SetBroadcastParallel4OpNodeIbn(JobBuilder* builder, const OpNode* node,
                                    const std::string& ibn) {
  OpBlobArg op_blob_arg;
  op_blob_arg.set_op_name(node->op().op_name());
  op_blob_arg.set_bn_in_op(ibn);
  builder->MutSbpParallel4Oba(op_blob_arg)->mutable_broadcast_parallel();
}

void SetBroadcastParallel4Consumers(JobBuilder* builder, const SequencePtr& sequence) {
  ForEachOutNodeConsumingSoleOut(sequence->GetLastNode(),
                                 [&](const OpNode* out_node, const std::string& ibn) {
                                   SetBroadcastParallel4OpNodeIbn(builder, out_node, ibn);
                                 });
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
    CHECK_JUST(GetDataParallelVariableAndNaiveSuccNode(node, IsAllowed, &nodes));
    if (nodes.empty()) { return; }
    const int64_t order = GetMinConsumerOrder(op_graph, nodes.back(), OpNode2Order);
    Handler(std::make_shared<const DataParallelNodeSequence>(std::move(nodes), order));
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

bool IsS0SignatureSupported(const OpNode* node) {
  if (node->op().input_bns().size() != 1 || node->op().output_bns().size() != 1) { return false; }
  SbpSignatureList list;
  auto LogicalBlobDesc4Ibn = [&](const std::string& bn) -> Maybe<const BlobDesc&> {
    return Maybe<const BlobDesc&>(node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi(bn)));
  };
  node->op().GetSbpSignaturesIf(LogicalBlobDesc4Ibn, node->parallel_desc(), &list);
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

Maybe<void> RewriteDistributedSplit(const OpGraph& op_graph, JobBuilder* builder) {
  const int64_t threshold = builder->job().job_conf().optimizer_placement_optimization_threshold();
  const auto IsAllowed = [threshold](const OpNode* n) -> bool {
    if (n->op().op_conf().has_variable_conf()) {
      const Shape shape(n->op().op_conf().variable_conf().shape());
      const int64_t parallel_num = n->parallel_desc().parallel_num();
      return shape.At(0) % parallel_num == 0 && shape.elem_cnt() >= threshold * parallel_num;
    } else {
      return IsS0SignatureSupported(n);
    }
  };
  const auto PlacementSequencesAsSplitParallel = [&](const ParallelDesc& pd,
                                                     std::vector<SequencePtr>&& sorted_sequences) {
    for (int64_t i = 0; i < sorted_sequences.size(); ++i) {
      const OpNode* var_node = sorted_sequences.at(i)->GetVariableNode();
      OperatorConf new_var_op_conf = var_node->op().op_conf();
      new_var_op_conf.mutable_variable_conf()->mutable_split_axis()->set_value(0);
      if (i != 0) {
        const std::string& prev_op_name =
            sorted_sequences.at(i - 1)->GetVariableNode()->op().op_name();
        new_var_op_conf.add_ctrl_in_op_name(prev_op_name);
      }
      builder->MutOpsOnlyOnce({new_var_op_conf});
      SetBroadcastParallel4Consumers(builder, sorted_sequences.at(i));
    }
  };
  ForEachParallelSortedNodeSequence(op_graph, IsAllowed, SequenceCompSortedByOrderAsc,
                                    PlacementSequencesAsSplitParallel);
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
          && ctx->job_desc().job_conf().has_optimizer_placement_optimization_mode())) {
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
      return Error::Unimplemented();
    }
  }
};

REGISTER_JOB_PASS("OptimizerPlacementOptimizationPass", OptimizerPlacementOptimizationPass);

}  // namespace

}  // namespace oneflow
