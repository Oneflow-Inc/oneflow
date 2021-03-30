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
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

// Do CheckpointingPass will use backward recomputation for sublinear memory cost.
class CheckpointingPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CheckpointingPass);
  CheckpointingPass() = default;
  ~CheckpointingPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

const std::string kCheckpointingFakeOpNamePrefix = "OneFlow-System-Checkpointing-Fake-Fw-Op_";
const std::string kCheckpointingBadOpName = "OneFlow-System-CheckpointPassBadEndOpName";

const Scope& Scope4OpNode(const OpNode* op_node) {
  int64_t scope_symbol_id = op_node->op().op_conf().scope_symbol_id();
  CHECK(Global<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id));
  return Global<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
}

bool IsForwardPassScope(const Scope& scope) {
  return scope.scope_proto().calculation_pass_name() == kForwardPass;
}

bool IsForwardPass7CheckpointingScope(const Scope& scope) {
  return IsForwardPassScope(scope) && scope.Bool("checkpointing");
}

void CollectAllCheckpointingOpsInForwardPass(
    const OpGraph& op_graph, HashMap<std::string, const OpNode*>* checkpointing_op_name2op_node) {
  // NOTE(chengcheng):
  //   ignore batch_norm ops because of recompute bn will repeat the calculation of 'm' and 'v'.
  //   in the future, we need to support the recomputation version of batch_norm which do NOT
  //   update forward variables.
  HashSet<std::string> ignore_op_type_names = {"normalization", "normalization_add_relu",
                                               "cudnn_fused_normalization_add_relu"};
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    if (ignore_op_type_names.find(op_conf.user_conf().op_type_name())
        != ignore_op_type_names.end()) {
      return;
    }
    if (IsForwardPass7CheckpointingScope(Scope4OpNode(op_node))) {
      CHECK(checkpointing_op_name2op_node->emplace(op_conf.name(), op_node).second);
    }
  });
}

void GenConnectedCheckpointingSubgraphs(
    const HashMap<std::string, const OpNode*>& checkpointing_op_name2op_node,
    std::vector<HashSet<const OpNode*>>* checkpointing_subgraphs) {
  HashSet<const OpNode*> visited_nodes;
  for (const auto& pair : checkpointing_op_name2op_node) {
    const OpNode* node = pair.second;
    if (visited_nodes.find(node) != visited_nodes.end()) { continue; }

    // new subgraph
    checkpointing_subgraphs->push_back(HashSet<const OpNode*>());
    CHECK(!checkpointing_subgraphs->empty());
    auto& subgraph = checkpointing_subgraphs->back();
    CHECK(subgraph.empty());

    // bfs search all node in checkpointing ops
    CHECK(visited_nodes.insert(node).second);
    std::queue<const OpNode*> queued_nodes;
    queued_nodes.push(node);
    while (!queued_nodes.empty()) {
      const OpNode* cur_node = queued_nodes.front();
      queued_nodes.pop();

      CHECK(subgraph.insert(cur_node).second);

      cur_node->ForEachNodeOnInOutEdge([&](const OpNode* next_node) {
        const std::string& next_op_name = next_node->op().op_name();
        if (checkpointing_op_name2op_node.find(next_op_name) != checkpointing_op_name2op_node.end()
            && cur_node->parallel_desc() == next_node->parallel_desc()
            && visited_nodes.find(next_node) == visited_nodes.end()) {
          queued_nodes.push(next_node);
          CHECK(visited_nodes.insert(next_node).second);
        }
      });
    }
  }
}

Maybe<void> CheckpointingPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  // step 1. collect all checkpointing ops in forwardpass.
  HashMap<std::string, const OpNode*> checkpointing_op_name2op_node;
  CollectAllCheckpointingOpsInForwardPass(op_graph, &checkpointing_op_name2op_node);
  if (checkpointing_op_name2op_node.empty()) { return Maybe<void>::Ok(); }

  // step 2. get all connected subgraphs in checkpointing ops.
  std::vector<HashSet<const OpNode*>> checkpointing_subgraphs;
  GenConnectedCheckpointingSubgraphs(checkpointing_op_name2op_node, &checkpointing_subgraphs);

  HashMap<const OpNode*, int32_t> op_node2order;
  int32_t order = 0;
  op_graph.TopoForEachNode([&](const OpNode* op_node) {
    CHECK(op_node2order.emplace(op_node, order).second);
    ++order;
  });

  // step 3. for each subgraphs:

  // NOTE(chengcheng):
  //   maybe a bw consumer will consume multi subgraph for recompute.
  //   so we need collect bw consumer between subgraphs, and update them in job builder only once.
  HashMap<std::string, OperatorConf> total_bw_consumers_op_name2conf;

  for (auto& subgraph : checkpointing_subgraphs) {
    // step 3.1 ignore this subgraph if there is no direct edge to backward pass op.
    HashSet<const OpNode*> bw_consumers;
    for (const OpNode* node : subgraph) {
      node->ForEachNodeOnOutEdge([&](const OpNode* out_node) {
        if (!IsForwardPassScope(Scope4OpNode(out_node))) {
          bw_consumers.insert(out_node);
          CHECK(subgraph.find(out_node) == subgraph.end());
        }
      });
    }
    if (bw_consumers.empty()) { continue; }

    HashMap<std::string, const OpNode*> subgraph_op_name2op_node;
    ParallelConf parallel_conf;
    for (const OpNode* node : subgraph) {
      subgraph_op_name2op_node.emplace(node->op().op_name(), node);
      parallel_conf = node->parallel_desc().parallel_conf();
    }

    // step 3.2 generate fake subgraph for recomputation
    HashMap<std::string, OperatorConf> fake_op_name2conf;
    HashSet<std::string> source_node_in_fake_subgraph;
    for (const OpNode* node : subgraph) {
      OperatorConf fake_op_conf = node->op().op_conf();
      std::string fake_op_name = kCheckpointingFakeOpNamePrefix + fake_op_conf.name();
      fake_op_conf.set_name(fake_op_name);
      const int64_t old_scope_symbol_id = fake_op_conf.scope_symbol_id();
      // update fake op conf scope from fw to bw
      const int64_t new_scope_symbol_id = JUST(NewScopeSymbolIdWihtModifier(
          old_scope_symbol_id, [](std::shared_ptr<cfg::ScopeProto> new_scope) {
            CHECK_EQ(new_scope->calculation_pass_name(), kForwardPass);
            new_scope->set_calculation_pass_name(kBackwardPass);
          }));
      fake_op_conf.set_scope_symbol_id(new_scope_symbol_id);

      auto* user_conf = fake_op_conf.mutable_user_conf();
      // change output lbns
      for (auto& pair : *(user_conf->mutable_output())) {
        auto& list_s = pair.second;
        for (int i = 0; i < list_s.s_size(); ++i) {
          std::string old_lbn = list_s.s(i);
          list_s.set_s(i, kCheckpointingFakeOpNamePrefix + old_lbn);
          // check valid
          LogicalBlobId old_lbi = GenLogicalBlobId(old_lbn);
          CHECK_EQ(node->op().op_conf().name(), old_lbi.op_name());
          CHECK_EQ(kCheckpointingFakeOpNamePrefix + old_lbi.op_name(), fake_op_name);
          std::string new_lbn = list_s.s(i);
          LogicalBlobId new_lbi = GenLogicalBlobId(new_lbn);
          CHECK_EQ(new_lbi.op_name(), fake_op_name);
          CHECK_EQ(old_lbi.blob_name(), new_lbi.blob_name());
        }
      }

      int32_t input_num = 0;
      // change input lbns if in subgraph
      for (auto& pair : *(user_conf->mutable_input())) {
        auto& list_s = pair.second;
        for (int i = 0; i < list_s.s_size(); ++i) {
          ++input_num;
          std::string old_lbn = list_s.s(i);
          LogicalBlobId old_lbi = GenLogicalBlobId(old_lbn);

          std::string old_input_op_name = old_lbi.op_name();
          if (subgraph_op_name2op_node.find(old_input_op_name) != subgraph_op_name2op_node.end()) {
            list_s.set_s(i, kCheckpointingFakeOpNamePrefix + old_lbn);
          } else {
            source_node_in_fake_subgraph.insert(fake_op_name);
          }
        }
      }
      if (input_num == 0) { source_node_in_fake_subgraph.insert(fake_op_name); }

      fake_op_name2conf.emplace(fake_op_name, fake_op_conf);
    }

    const OpNode* first_bw_consumer = nullptr;
    int32_t first_bw_order = std::numeric_limits<int32_t>::max();
    // step 3.3 change bw consumers input from subgraph to fake subgraph
    for (const OpNode* node : bw_consumers) {
      std::string bw_consumer_name = node->op().op_name();
      OperatorConf bw_consumer_op_conf;
      // NOTE(chengcheng):
      //   reuse bw conumer op conf if it has been existed in map.
      if (total_bw_consumers_op_name2conf.find(bw_consumer_name)
          != total_bw_consumers_op_name2conf.end()) {
        bw_consumer_op_conf = total_bw_consumers_op_name2conf.at(bw_consumer_name);
      } else {
        bw_consumer_op_conf = node->op().op_conf();
      }
      CHECK_EQ(bw_consumer_name, bw_consumer_op_conf.name());

      auto* user_conf = bw_consumer_op_conf.mutable_user_conf();
      // change input lbns if in subgraph
      for (auto& pair : *(user_conf->mutable_input())) {
        auto& list_s = pair.second;
        for (int i = 0; i < list_s.s_size(); ++i) {
          std::string old_lbn = list_s.s(i);
          LogicalBlobId old_lbi = GenLogicalBlobId(old_lbn);

          std::string old_input_op_name = old_lbi.op_name();
          if (subgraph_op_name2op_node.find(old_input_op_name) != subgraph_op_name2op_node.end()) {
            list_s.set_s(i, kCheckpointingFakeOpNamePrefix + old_lbn);
          }
        }
      }

      // NOTE(chengcheng):
      //   emplace maybe repeated, so do not check the return value
      total_bw_consumers_op_name2conf.emplace(bw_consumer_name, bw_consumer_op_conf);

      CHECK(op_node2order.find(node) != op_node2order.end());
      int32_t this_order = op_node2order.at(node);
      if (this_order < first_bw_order) {
        first_bw_consumer = node;
        first_bw_order = this_order;
      }
    }

    // step 3.4 add control edge from End Op to all source node in fake subgraph
    CHECK(first_bw_consumer != nullptr);
    std::string end_op_name = kCheckpointingBadOpName;
    int32_t end_order = -1;
    first_bw_consumer->ForEachNodeOnInEdge([&](const OpNode* end_node) {
      CHECK(op_node2order.find(end_node) != op_node2order.end());
      int32_t this_order = op_node2order.at(end_node);
      if (this_order > end_order) {
        end_order = this_order;
        end_op_name = end_node->op().op_name();
      }
    });
    CHECK_NE(end_order, -1);
    CHECK_NE(end_op_name, kCheckpointingBadOpName);
    CHECK_LT(end_order, first_bw_order);
    for (const auto& source_op_name : source_node_in_fake_subgraph) {
      fake_op_name2conf.at(source_op_name).add_ctrl_in_op_name(end_op_name);
    }

    // step 3.5 add fake subgraph ops to job builder
    std::vector<OperatorConf> fake_op_confs;
    for (auto& pair : fake_op_name2conf) { fake_op_confs.push_back(pair.second); }
    job_builder->AddOps(parallel_conf, fake_op_confs);
  }

  // step 4. update bw consumers in job builder only once
  std::vector<OperatorConf> total_bw_consumer_op_confs;
  for (auto& pair : total_bw_consumers_op_name2conf) {
    total_bw_consumer_op_confs.push_back(pair.second);
  }
  job_builder->MutOpsOnlyOnce(total_bw_consumer_op_confs);

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("CheckpointingPass", CheckpointingPass);

}  // namespace oneflow
