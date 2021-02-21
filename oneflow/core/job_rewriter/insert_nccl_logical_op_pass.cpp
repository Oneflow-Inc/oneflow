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
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

// Do InsertNcclLogicalOpPass will use backward recomputation for sublinear memory cost.
class InsertNcclLogicalOpPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InsertNcclLogicalOpPass);
  InsertNcclLogicalOpPass() = default;
  ~InsertNcclLogicalOpPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return Global<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream();
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

const std::string kNcclLogicalOpNamePrefix = "System-NCCL-Logical";

void FindMaxConnectedSubgraphForGpuExecOrder(HashSet<const OpNode*>* ret, const OpGraph& op_graph,
                                             const std::vector<const OpNode*>& order) {
  HashSet<const OpNode*> visited;

  for (const OpNode* seed_node : order) {
    if (visited.find(seed_node) != visited.end()) { continue; }
    CHECK(visited.insert(seed_node).second);
    const ParallelDesc& seed_parallel_desc = seed_node->parallel_desc();
    // NOTE(chengcheng): ONLY consider GPU op and parallel num > 1.
    if (seed_parallel_desc.device_type() != DeviceType::kGPU) { continue; }
    if (seed_parallel_desc.parallel_num() <= 1) { continue; }
    // NODE(chengcheng): Exclude op that change the time shape.
    //   like pack/unpack, repeat/acc, etc.
    if (!seed_node->IsTimeShapeIdentity()) { continue; }

    HashSet<const OpNode*> this_subgraph;
    std::queue<const OpNode*> queued_nodes;
    queued_nodes.push(seed_node);
    while (!queued_nodes.empty()) {
      const OpNode* cur_node = queued_nodes.front();
      queued_nodes.pop();

      CHECK(cur_node->parallel_desc() == seed_parallel_desc);
      CHECK(this_subgraph.insert(cur_node).second);

      cur_node->ForEachNodeOnInOutEdge([&](const OpNode* next_node) {
        if (visited.find(next_node) == visited.end()
            && next_node->parallel_desc() == seed_parallel_desc
            && next_node->IsTimeShapeIdentity()) {
          CHECK(visited.insert(next_node).second);
          queued_nodes.push(next_node);
        }
      });
    }

    if (this_subgraph.size() > ret->size()) { ret->swap(this_subgraph); }
  }
}

bool TryBuildNcclLogicalOpConf(OperatorConf* ret, const OpNode* src_node, const OpNode* dst_node,
                               const LogicalBlobId& lbi) {
  const int64_t scope_symbol_id = src_node->op().op_conf().scope_symbol_id();
  const std::string lbn = GenLogicalBlobName(lbi);
  const SbpParallel& src_sbp = src_node->SbpParallel4Lbi(lbi);
  const SbpParallel& dst_sbp = dst_node->SbpParallel4Lbi(lbi);
  const BlobDesc& logical_blob_desc = src_node->LogicalBlobDesc4Lbi(lbi);
  const ParallelDesc& parallel_desc = src_node->parallel_desc();

  // NOTE(chengcheng): nccl donot support dynamic shape.
  if (logical_blob_desc.is_dynamic()) { return false; }
  CHECK_GT(logical_blob_desc.shape().elem_cnt(), 0);
  CHECK_GT(logical_blob_desc.shape().NumAxes(), 0);
  if (src_sbp.has_partial_sum_parallel() && dst_sbp.has_broadcast_parallel()) {
    // P2B : AllReduce
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-P2B-" + NewUniqueId())
               .Op("_nccl_logical_all_reduce")
               .Input("in", lbn)
               .Output("out")
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if ((logical_blob_desc.shape().At(0) % parallel_desc.parallel_num() == 0)
             && (src_sbp.has_partial_sum_parallel() && dst_sbp.has_split_parallel())
             && (dst_sbp.split_parallel().axis() == 0)) {
    // P2S : ReduceScatter
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-P2S-" + NewUniqueId())
               .Op("_nccl_logical_reduce_scatter")
               .Input("in", lbn)
               .Output("out")
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if ((logical_blob_desc.shape().At(0) % parallel_desc.parallel_num() == 0)
             && (src_sbp.has_split_parallel() && dst_sbp.has_broadcast_parallel())
             && (src_sbp.split_parallel().axis() == 0)) {
    // S2B : AllGather
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-S2B-" + NewUniqueId())
               .Op("_nccl_logical_all_gather")
               .Input("in", lbn)
               .Output("out")
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if ((src_sbp.has_split_parallel() && dst_sbp.has_split_parallel())
             && (src_sbp.split_parallel().axis() != dst_sbp.split_parallel().axis())
             && (logical_blob_desc.shape().At(src_sbp.split_parallel().axis())
                     % parallel_desc.parallel_num()
                 == 0)
             && (logical_blob_desc.shape().At(dst_sbp.split_parallel().axis())
                     % parallel_desc.parallel_num()
                 == 0)) {
    // S2S : All2All
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-S2S-" + NewUniqueId())
               .Op("_nccl_logical_s2s")
               .Input("in", lbn)
               .Output("out")
               .Attr<int64_t>("in_split_axis", src_sbp.split_parallel().axis())
               .Attr<int64_t>("out_split_axis", dst_sbp.split_parallel().axis())
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  }
  return false;
}

bool ReverseOrderInsertNcclLogicalOps() {
  return Global<ResourceDesc, ForSession>::Get()->resource().disable_group_boxing_by_dst_parallel();
}

void InsertNcclLogicalOpsAsCloseAsPossibleToSrcNode(
    HashMap<std::string, OperatorConf>* subgraph_op_name2conf, HashSet<std::string>* mut_op_names,
    std::vector<OperatorConf>* nccl_op_confs, const std::vector<const OpNode*>& subgraph_order,
    const HashMap<const OpNode*, int64_t>& node2order) {
  for (const OpNode* src_node : subgraph_order) {
    const std::string& src_op_name = src_node->op().op_name();
    for (const OpEdge* op_edge : src_node->out_edges()) {
      const OpNode* dst_node = op_edge->dst_node();
      const std::string& dst_op_name = dst_node->op().op_name();
      CHECK(src_node != dst_node);
      if (subgraph_op_name2conf->find(dst_op_name) == subgraph_op_name2conf->end()) {
        // NOTE(chengcheng): child node is not in this subgraph.
        continue;
      }
      for (const LogicalBlobId& lbi : op_edge->lbis()) {
        OperatorConf nccl_op;
        if (!TryBuildNcclLogicalOpConf(&nccl_op, src_node, dst_node, lbi)) { continue; }
        mut_op_names->insert(dst_op_name);
        // insert nccl op
        user_op::UserOpConfWrapper nccl_op_wrapper(nccl_op);
        for (const std::string& ibn : op_edge->lbi2ibns().at(lbi)) {
          std::string old_lbn = ReplaceInputLbnInOpCustomizedConf(
              &subgraph_op_name2conf->at(dst_op_name), ibn, nccl_op_wrapper.output("out", 0));
        }

        if (nccl_op_confs->size() >= 1) {
          // NOTE(chengcheng): MUST add ctrl edge between nccl ops for 1 src node insert multi-nccl
          const std::string& pre_nccl_op_name = nccl_op_confs->at(nccl_op_confs->size() - 1).name();
          nccl_op.add_ctrl_in_op_name(pre_nccl_op_name);
        }

        // NOTE(chengcheng): src_node MUST not the last node in subgraph, find the next op
        int64_t src_order = node2order.at(src_node);
        CHECK(src_order + 1 < subgraph_order.size());
        const std::string& next_op_name = subgraph_order.at(src_order + 1)->op().op_name();
        if (dst_op_name != next_op_name) {
          // NOTE(chengcheng): MUST add ctrl edge for strict exec order
          subgraph_op_name2conf->at(next_op_name).add_ctrl_in_op_name(nccl_op.name());
          mut_op_names->insert(next_op_name);
        }

        if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
          std::cout << "cc_debug_log: insert nccl op from: [" << src_op_name << "](" << src_order
                    << ")->[" << dst_op_name << "](" << node2order.at(dst_node) << ") and before: ["
                    << next_op_name << "](" << src_order + 1 << ")\n";
        }
        nccl_op_confs->push_back(nccl_op);
      }
    }
  }
}

void InsertNcclLogicalOpsAsCloseAsPossibleToDstNode(
    HashMap<std::string, OperatorConf>* subgraph_op_name2conf, HashSet<std::string>* mut_op_names,
    std::vector<OperatorConf>* nccl_op_confs, const std::vector<const OpNode*>& subgraph_order,
    const HashMap<const OpNode*, int64_t>& node2order) {
  for (const OpNode* dst_node : subgraph_order) {
    const std::string& dst_op_name = dst_node->op().op_name();
    for (const OpEdge* op_edge : dst_node->in_edges()) {
      const OpNode* src_node = op_edge->src_node();
      const std::string& src_op_name = src_node->op().op_name();
      CHECK(src_node != dst_node);
      if (subgraph_op_name2conf->find(src_op_name) == subgraph_op_name2conf->end()) {
        // NOTE(chengcheng): parent node is not in this subgraph.
        continue;
      }
      for (const LogicalBlobId& lbi : op_edge->lbis()) {
        OperatorConf nccl_op;
        // builde nccl op
        if (!TryBuildNcclLogicalOpConf(&nccl_op, src_node, dst_node, lbi)) { continue; }
        mut_op_names->insert(dst_op_name);
        // insert nccl op
        user_op::UserOpConfWrapper nccl_op_wrapper(nccl_op);
        for (const std::string& ibn : op_edge->lbi2ibns().at(lbi)) {
          std::string old_lbn = ReplaceInputLbnInOpCustomizedConf(
              &subgraph_op_name2conf->at(dst_op_name), ibn, nccl_op_wrapper.output("out", 0));
          CHECK(old_lbn == GenLogicalBlobName(lbi));
        }

        // add necessary ctrl edge for strict order
        if (nccl_op_confs->size() >= 1) {
          // NOTE(chengcheng): MUST add ctrl edge between nccl ops for 1 dst node insert multi-nccl
          const std::string& pre_nccl_op_name = nccl_op_confs->at(nccl_op_confs->size() - 1).name();
          nccl_op.add_ctrl_in_op_name(pre_nccl_op_name);
        }

        // NOTE(chengcheng): dst_node MUST not the first node in subgraph, find the Immediately
        //   previous op of dst_node.
        int64_t dst_order = node2order.at(dst_node);
        CHECK_GT(dst_order, 0);
        const std::string& pre_op_name = subgraph_order.at(dst_order - 1)->op().op_name();
        if (src_op_name != pre_op_name) {
          // NOTE(chengcheng): MUST add ctrl edge for strict exec order
          nccl_op.add_ctrl_in_op_name(pre_op_name);
        }

        if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
          std::cout << "cc_debug_log: insert nccl op from: [" << src_op_name << "]("
                    << node2order.at(src_node) << ")->[" << dst_op_name << "](" << dst_order
                    << ") and after: [" << pre_op_name << "](" << dst_order - 1 << ")\n";
        }
        nccl_op_confs->push_back(nccl_op);
      }
    }
  }
}

Maybe<void> InsertNcclLogicalOpPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  auto OpGraphForEachInDataAndCtrlNode = [&](OpNode* node,
                                             const std::function<void(OpNode*)>& Handler) {
    op_graph.ForEachDataAndCtrlInNode(node, Handler);
  };
  auto OpGraphForEachOutDataAndCtrlNode = [&](OpNode* node,
                                              const std::function<void(OpNode*)>& Handler) {
    op_graph.ForEachDataAndCtrlOutNode(node, Handler);
  };

  std::vector<const OpNode*> ordered_op_nodes;
  op_graph.TopoForEachNode(op_graph.DataOrCtrlSourceNodes(), OpGraphForEachInDataAndCtrlNode,
                           OpGraphForEachOutDataAndCtrlNode,
                           [&](const OpNode* node) { ordered_op_nodes.push_back(node); });

  HashSet<const OpNode*> subgraph;
  FindMaxConnectedSubgraphForGpuExecOrder(&subgraph, op_graph, ordered_op_nodes);
  if (subgraph.size() <= 1) { return Maybe<void>::Ok(); }

  std::vector<const OpNode*> subgraph_order;
  HashMap<const OpNode*, int64_t> node2order;
  for (const OpNode* this_node : ordered_op_nodes) {
    if (subgraph.find(this_node) != subgraph.end()) {
      subgraph_order.push_back(this_node);
      node2order.emplace(this_node, subgraph_order.size() - 1);
    }
  }
  CHECK_EQ(subgraph.size(), subgraph_order.size());

  HashSet<std::string> mut_op_names;
  const OpNode* first_node = subgraph_order.at(0);
  HashMap<std::string, OperatorConf> subgraph_op_name2conf;
  subgraph_op_name2conf.emplace(first_node->op().op_name(), first_node->op().op_conf());
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  for (int32_t i = 1; i < subgraph_order.size(); ++i) {
    const OpNode* this_node = subgraph_order.at(i);
    const OpNode* pre_node = subgraph_order.at(i - 1);
    const std::string& this_op_name = this_node->op().op_name();
    const std::string& pre_op_name = pre_node->op().op_name();
    CHECK(subgraph_op_name2conf.emplace(this_op_name, this_node->op().op_conf()).second);
    // build control edge if need.
    if (!IsReachable(pre_op_name, this_op_name)) {
      subgraph_op_name2conf.at(this_op_name).add_ctrl_in_op_name(pre_op_name);
      mut_op_names.insert(this_op_name);
    }
  }

  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    std::cout << "cc_debug_log: Try insert nccl logical ops into job: "
              << job_builder->job().job_conf().job_name() << ". Begin...\n";
  }

  std::vector<OperatorConf> nccl_op_confs;
  if (ReverseOrderInsertNcclLogicalOps()) {
    InsertNcclLogicalOpsAsCloseAsPossibleToDstNode(&subgraph_op_name2conf, &mut_op_names,
                                                   &nccl_op_confs, subgraph_order, node2order);
  } else {
    InsertNcclLogicalOpsAsCloseAsPossibleToSrcNode(&subgraph_op_name2conf, &mut_op_names,
                                                   &nccl_op_confs, subgraph_order, node2order);
  }

  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    std::cout << "cc_debug_log: Try insert nccl logical ops into job: "
              << job_builder->job().job_conf().job_name() << ". ...End\n\n";
  }

  std::vector<OperatorConf> mut_op_confs;
  for (const std::string& mut_op_name : mut_op_names) {
    mut_op_confs.push_back(subgraph_op_name2conf.at(mut_op_name));
  }
  job_builder->MutOpsOnlyOnce(mut_op_confs);
  job_builder->AddOps(first_node->parallel_desc().parallel_conf(), nccl_op_confs);

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("InsertNcclLogicalOpPass", InsertNcclLogicalOpPass);

}  // namespace oneflow
