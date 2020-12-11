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
#include <functional>
#include "oneflow/core/graph/compute_graph.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

std::string ComputeNode::VisualStr() const { return op().op_name(); }

Maybe<int64_t> ComputeNode::GetParallelDescSymbolId() const {
  return scope().GetParallelDescSymbolId(op().op_conf());
}

Maybe<void> ComputeNode::Init(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  scope_ = JUST(Global<vm::SymbolStorage<Scope>>::Get()->MaybeGetPtr(op_conf.scope_symbol_id()));
  parallel_desc_ = JUST(scope_->GetParallelDescPtr(op_conf));
  op_ = ConstructOp(op_conf, parallel_desc_->device_type(), JUST(scope_->job_desc()));
  return Maybe<void>::Ok();
}

Maybe<void> ComputeGraph::Init(const Job& job) {
  JUST(InitNodes(job));
  JUST(InitEdges(job));
  get_dot_end_ = []() -> Maybe<std::string> { return std::make_shared<std::string>(); };
  return Maybe<void>::Ok();
}

Maybe<void> ComputeGraph::InitNodes(const Job& job) {
  for (const auto& op_conf : job.net().op()) {
    op_names_.push_back(op_conf.name());
    auto* node = JUST(ComputeNode::UnsafeNew(op_conf));
    AddAllocatedNode(node);
    CHECK_OR_RETURN(op_name2node_.emplace(node->op().op_name(), node).second);
  }
  for (const auto& op_conf : job.net().op()) {
    const auto* node = &JUST(Node4OpName(op_conf.name()));
    for (const auto& producer_op_name : op_conf.ctrl_in_op_name()) {
      auto* consumer_nodes = &producer_op_name2ctrl_consumer_nodes_[producer_op_name];
      consumer_nodes->push_back(node);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ComputeGraph::InitEdges(const Job& job) {
  JUST(MaybeForEachNode([&](const ComputeNode* consumer_node) -> Maybe<void> {
    HashSet<LogicalBlobId> connected_lbis;
    for (const auto& ibn : consumer_node->op().input_bns()) {
      const LogicalBlobId& lbi = consumer_node->op().BnInOp2Lbi(ibn);
      if (connected_lbis.count(lbi) > 0) { continue; }
      auto* producer_node = JUST(MutNode4OpName(lbi.op_name()));
      Connect(producer_node, const_cast<ComputeEdge*>(NewEdge()),
              const_cast<ComputeNode*>(consumer_node));
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

void ComputeGraph::ForEachDataAndCtrlInNode(
    const ComputeNode* node, const std::function<void(const ComputeNode*)>& Handler) const {
  HashSet<const ComputeNode*> visited_nodes;
  const auto& HandleOnce = [&](const ComputeNode* node) {
    if (visited_nodes.count(node) > 0) { return; }
    Handler(node);
    visited_nodes.insert(node);
  };
  node->ForEachNodeOnInEdge(HandleOnce);
  for (const auto& ctrl_in_op_name : node->op().op_conf().ctrl_in_op_name()) {
    HandleOnce(&CHECK_JUST(Node4OpName(ctrl_in_op_name)));
  }
}

void ComputeGraph::ForEachDataAndCtrlOutNode(
    const ComputeNode* node, const std::function<void(const ComputeNode*)>& Handler) const {
  HashSet<const ComputeNode*> visited_nodes;
  const auto& HandleOnce = [&](const ComputeNode* node) {
    if (visited_nodes.count(node) > 0) { return; }
    Handler(node);
    visited_nodes.insert(node);
  };
  node->ForEachNodeOnOutEdge(HandleOnce);
  const auto& nodes = MapAtOrDefault(producer_op_name2ctrl_consumer_nodes_, node->op().op_name());
  for (const auto* consumer_node : nodes) { HandleOnce(consumer_node); }
}

Maybe<void> ComputeGraph::InitTopoOrderValue() {
  using namespace std::placeholders;
  int64_t topo_order_value = 0;
  const auto& UpdateTopoOrderValue = [&](const ComputeNode* node) -> Maybe<void> {
    auto* mut_node = const_cast<ComputeNode*>(node);
    CHECK_ISNULL_OR_RETURN(mut_node->topo_order_value_.get())
        << "Do not initialize topo_order_value twice";
    mut_node->topo_order_value_.reset(new int64_t());
    *mut_node->topo_order_value_ = ++topo_order_value;
    return Maybe<void>::Ok();
  };
  JUST(MaybeTopoForEachNode(
      source_nodes(), std::bind(&ComputeGraph::ForEachDataAndCtrlInNode, this, _1, _2),
      std::bind(&ComputeGraph::ForEachDataAndCtrlOutNode, this, _1, _2), UpdateTopoOrderValue));
  return Maybe<void>::Ok();
}

Maybe<ComputeNode*> ComputeGraph::MutNode4OpName(const std::string& op_name) {
  return MapAt(op_name2node_, op_name);
}

Maybe<void> ComputeGraph::ForEachComputeNode(
    const std::function<Maybe<void>(const ComputeNode&)>& DoEach) const {
  for (const auto& op_name : op_names_) { JUST(DoEach(JUST(Node4OpName(op_name)))); }
  return Maybe<void>::Ok();
}

Maybe<void> ComputeGraph::TopoForEachComputeNode(
    const std::function<Maybe<void>(const ComputeNode&)>& DoEach) const {
  return MaybeTopoForEachNode(
      [&](const ComputeNode* node) -> Maybe<void> { return DoEach(*node); });
}

Maybe<void> ComputeGraph::WithDotEndGetter(const std::function<Maybe<std::string>()>& get_dot_end,
                                           const std::function<Maybe<void>()>& Do) {
  get_dot_end_ = get_dot_end;
  JUST(Do());
  get_dot_end_ = []() -> Maybe<std::string> { return std::make_shared<std::string>(); };
  return Maybe<void>::Ok();
}

Maybe<std::string> ComputeGraph::VirtualDotEnd() const { return get_dot_end_(); }

}  // namespace oneflow
