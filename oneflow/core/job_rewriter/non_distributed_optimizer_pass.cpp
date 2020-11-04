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

ParallelConf NonDistributedParallelConf4ParallelId(const ParallelDesc& pd,
                                                   const int64_t parallel_id) {
  std::string device_name;
  device_name += std::to_string(CHECK_JUST(pd.MachineId4ParallelId(parallel_id)));
  device_name += ":";
  device_name += std::to_string(CHECK_JUST(pd.DeviceId4ParallelId(parallel_id)));
  ParallelConf parallel_conf;
  *parallel_conf.mutable_device_name()->Add() = device_name;
  if (pd.device_type() == DeviceType::kGPU) {
    parallel_conf.set_device_tag("gpu");
  } else if (pd.device_type() == DeviceType::kCPU) {
    parallel_conf.set_device_tag("cpu");
  } else {
    UNIMPLEMENTED();
  }
  return parallel_conf;
}

class NonDistributedOptimizerPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonDistributedOptimizerPass);
  NonDistributedOptimizerPass() = default;
  ~NonDistributedOptimizerPass() = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain() && ctx.job_desc().enable_non_distributed_optimizer();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> NonDistributedOptimizerPass::Apply(const OpGraph& op_graph, JobBuilder* builder) const {
  HashMap<ParallelDesc, HashMap<const OpNode*, std::vector<const OpNode*>>> pd2last_node2node_seqs;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  HashMap<const OpNode*, int64_t> last_node2model_size;
  HashMap<const OpNode*, int64_t> op_node2topo_order;
  HashMap<ParallelDesc, std::vector<const OpNode*>> pd2last_nodes;
  HashMap<const OpNode*, int64_t> last_node2parallel_id;
  HashMap<const OpNode*, int64_t> last_node2order;
  HashMap<const OpNode*, const OpNode*> last_node2var_node;
  int64_t node_cnt = 0;
  op_graph.TopoForEachNode([&](const OpNode* node) {
    op_node2topo_order[node] = node_cnt;
    node_cnt += 1;
  });
  op_graph.ForEachNode([&](const OpNode* node) {
    if (!node->op().op_conf().has_variable_conf()) { return; }
    std::vector<const OpNode*> op_seq_without_batch_dim;
    const OpNode* cur_node = node;
    while (cur_node != nullptr) {
      if (cur_node->parallel_desc().device_type() != DeviceType::kGPU) { break; }
      if (cur_node != node) {
        if (cur_node->in_edges().size() > 1) { break; }
        if (cur_node->op().input_bns().size() != 1) { break; }
        const std::string& sole_ibn = cur_node->op().input_bns().Get(0);
        if (!cur_node->SbpParallel4BnInOp(sole_ibn).has_broadcast_parallel()) { break; }
      }
      if (cur_node->op().output_bns().size() != 1) { break; }
      const std::string& sole_obn = cur_node->op().output_bns().Get(0);
      if (!cur_node->SbpParallel4BnInOp(sole_obn).has_broadcast_parallel()) { break; }
      const auto& lbi = cur_node->op().BnInOp2Lbi(sole_obn);
      if (CHECK_JUST(cur_node->BatchAxis4Lbi(lbi))->has_value()) { break; }
      op_seq_without_batch_dim.push_back(cur_node);
      if (cur_node->out_edges().size() == 1) {
        cur_node = cur_node->SoleOutEdge()->dst_node();
      } else {
        cur_node = nullptr;
      }
    }
    if (op_seq_without_batch_dim.empty()) { return; }
    const OpNode* last_node = op_seq_without_batch_dim.back();
    const ParallelDesc& pd = last_node->parallel_desc();
    pd2last_node2node_seqs[pd][last_node] = op_seq_without_batch_dim;
    const OpNode* var_node = op_seq_without_batch_dim.front();
    CHECK(var_node->op().op_conf().has_variable_conf());
    last_node2var_node[last_node] = var_node;
    int64_t min_consumer_topo_order = node_cnt;
    last_node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
      op_node2op_conf.emplace(dst, dst->op().op_conf());
      min_consumer_topo_order = std::min(min_consumer_topo_order, op_node2topo_order.at(dst));
    });
    last_node2model_size[last_node] = GetSoleOutBlobSize(node);
    last_node2order[last_node] = min_consumer_topo_order;
  });
  for (const auto& pair : pd2last_node2node_seqs) {
    const ParallelDesc& pd = pair.first;
    if (pd.parallel_num() <= 1) { continue; }
    std::vector<int64_t> parallel_id2size(pd.parallel_num(), 0);

    std::vector<std::pair<const OpNode*, int64_t>> last_node_out_size_pairs;
    const auto& last_node2node_seqs = pair.second;
    for (const auto& last_node7node_seqs : last_node2node_seqs) {
      const OpNode* last_node = last_node7node_seqs.first;
      last_node_out_size_pairs.emplace_back(last_node, last_node2model_size.at(last_node));
    }
    std::sort(
        last_node_out_size_pairs.begin(), last_node_out_size_pairs.end(),
        [&](const std::pair<const OpNode*, int64_t>& lhs,
            const std::pair<const OpNode*, int64_t>& rhs) { return lhs.second > rhs.second; });
    for (const auto& last_node7out_size : last_node_out_size_pairs) {
      const auto it = std::min_element(parallel_id2size.cbegin(), parallel_id2size.cend());
      const int64_t min_parallel_id = std::distance(parallel_id2size.cbegin(), it);
      last_node2parallel_id[last_node7out_size.first] = min_parallel_id;
      parallel_id2size[min_parallel_id] += last_node7out_size.second;
    }
    for (const auto& last_node7node_seqs : last_node2node_seqs) {
      const OpNode* last_node = last_node7node_seqs.first;
      const int64_t& parallel_id = last_node2parallel_id.at(last_node);
      const ParallelConf parallel_conf = NonDistributedParallelConf4ParallelId(pd, parallel_id);
      for (const OpNode* node : last_node7node_seqs.second) {
        builder->MutParallelConfOnlyOnce(node->op().op_name(), parallel_conf);
      }
      ParallelDesc new_pd(parallel_conf);
      pd2last_nodes[new_pd].push_back(last_node);
    }
  }
  for (auto& pair : pd2last_nodes) {
    std::vector<const OpNode*>* last_nodes = &pair.second;
    std::sort(last_nodes->begin(), last_nodes->end(), [&](const OpNode* lhs, const OpNode* rhs) {
      return last_node2order.at(lhs) < last_node2order.at(rhs);
    });
    FOR_RANGE(int64_t, i, 1, last_nodes->size()) {
      const OpNode* cur_var_node = last_node2var_node.at(last_nodes->at(i));
      OperatorConf cur_var_conf(cur_var_node->op().op_conf());
      const OpNode* prev_var_node = last_node2var_node.at(last_nodes->at(i - 1));
      cur_var_conf.add_ctrl_in_op_name(prev_var_node->op().op_name());
      builder->MutOpsOnlyOnce({cur_var_conf});
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("NonDistributedOptimizerPass", NonDistributedOptimizerPass);

}  // namespace

}  // namespace oneflow
