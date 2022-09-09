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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/common/env_var/debug_mode.h"

namespace oneflow {

namespace {

class LogicalChainPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalChainPass);
  LogicalChainPass() = default;
  ~LogicalChainPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const { return EnableLogicalChain(); }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

bool IsBreakpointOpNode(const OpNode* node) {
  // NOTE(chengcheng): breakpoint op is special which CANNOT merge in chain such as:
  //   variable, tick, repeat/acc/pack/unpack change timeshape
  const Operator& op = node->op();
  const OperatorConf& op_conf = op.op_conf();

  // TODO(chengcheng): filter ops which has special type
  // TODO(chengcheng): get stream by op type
  if (op_conf.has_variable_conf() /* varialbe */
      || op_conf.has_tick_conf() || op_conf.has_device_tick_conf()
      || op_conf.has_src_subset_tick_conf() || op_conf.has_dst_subset_tick_conf()
      || op_conf.has_source_tick_conf() || op_conf.has_sink_tick_conf()
      || op_conf.has_acc_tick_conf() || op_conf.has_critical_section_wait_tick_conf()
      || op_conf.has_critical_section_callback_tick_conf()                          /* tick */
      || op_conf.has_input_conf() || op_conf.has_output_conf()                      /* io */
      || op_conf.has_wait_and_send_ids_conf() || op_conf.has_callback_notify_conf() /* ctrl */
      || op_conf.has_image_decoder_random_crop_resize_conf() /* gpu decode */) {
    return true;
  }

  if (op_conf.has_user_conf()) {
    const std::string& user_type_name = op_conf.user_conf().op_type_name();
    if (user_type_name == "repeat" || user_type_name == "pack" || user_type_name == "unpack"
        || user_type_name == "identity_buffer" || user_type_name == "copy_h2d"
        || user_type_name == "copy_d2h" || user_type_name == "acc_ctrl_tick") {
      return true;
    }
  }
  return false;
}

bool IsAccOpNode(const OpNode* node) {
  return node->op().op_conf().has_user_conf()
         && node->op().op_conf().user_conf().op_type_name() == "acc";
}

bool IsRepeatOpNode(const OpNode* node) {
  return node->op().op_conf().has_user_conf()
         && node->op().op_conf().user_conf().op_type_name() == "repeat";
}

std::shared_ptr<const Shape> GetOpNodeFastestTimeShape(const OpNode* op_node) {
  return CHECK_JUST(op_node->op().GetInputOutputFastestTimeShape());
}

std::shared_ptr<const Shape> GetOpNodeInputTimeShape(const OpNode* op_node) {
  return CHECK_JUST(op_node->op().GetInputBlobFastestTimeShape());
}

bool SharedPtrShapeEqual(const std::shared_ptr<const Shape>& lhs,
                         const std::shared_ptr<const Shape>& rhs) {
  return (*lhs) == (*rhs);
}

bool NeedInsertBoxingBetweenOpNodes(const OpNode* a_node, const OpNode* b_node,
                                    const OpEdge* edge) {
  CHECK(a_node != b_node && (edge->src_node() == a_node || edge->src_node() == b_node)
        && (edge->dst_node() == a_node || edge->dst_node() == b_node));
  if (a_node->parallel_desc().parallel_num() > 1) {
    for (const auto& lbi : edge->lbis()) {
      if (a_node->NdSbp4Lbi(lbi) != b_node->NdSbp4Lbi(lbi)) { return true; }
    }
  }
  return false;
}

void GetLogicalChainsWithTimeShape(std::vector<HashSet<const OpNode*>>* ret,
                                   const std::vector<const OpNode*>& order,
                                   const std::shared_ptr<const Shape>& seed_time_shape) {
  HashSet<const OpNode*> visited;
  for (const OpNode* seed_node : order) {
    if (visited.find(seed_node) != visited.end()) { continue; }
    CHECK(visited.insert(seed_node).second);
    const ParallelDesc& seed_parallel_desc = seed_node->parallel_desc();
    // TODO(chengcheng): support cpu chain.
    if (seed_parallel_desc.device_type() == DeviceType::kCPU) { continue; }
    if (!SharedPtrShapeEqual(GetOpNodeFastestTimeShape(seed_node), seed_time_shape)) { continue; }
    if (IsBreakpointOpNode(seed_node)) { continue; }

    HashSet<const OpNode*> this_subgraph;
    std::queue<const OpNode*> queued_nodes;

    queued_nodes.push(seed_node);
    while (!queued_nodes.empty()) {
      const OpNode* cur_node = queued_nodes.front();
      queued_nodes.pop();

      CHECK(cur_node->parallel_desc().EqualsIgnoringHierarchy(seed_parallel_desc));
      CHECK(this_subgraph.insert(cur_node).second);

      auto SearchToNextNode = [&](const OpNode* cur_node, const OpNode* next_node,
                                  const OpEdge* edge) {
        if (visited.find(next_node) == visited.end() && (!IsBreakpointOpNode(next_node))
            && next_node->parallel_desc().EqualsIgnoringHierarchy(seed_parallel_desc)
            && SharedPtrShapeEqual(GetOpNodeFastestTimeShape(next_node), seed_time_shape)
            && !NeedInsertBoxingBetweenOpNodes(cur_node, next_node, edge)) {
          CHECK(visited.insert(next_node).second);
          queued_nodes.push(next_node);
        }
      };

      for (const OpEdge* in_edge : cur_node->in_edges()) {
        SearchToNextNode(cur_node, in_edge->src_node(), in_edge);
      }
      for (const OpEdge* out_edge : cur_node->out_edges()) {
        SearchToNextNode(cur_node, out_edge->dst_node(), out_edge);
      }
    }

    if (this_subgraph.size() > 1) {
      ret->emplace_back(HashSet<const OpNode*>());
      ret->back().swap(this_subgraph);
    }
  }
}

struct LogicalChain {
  int64_t logical_chain_id;
  std::vector<const OpNode*> ordered_op_nodes;
  LogicalChain() : logical_chain_id(-1) {}
};

struct PlacementLogicalChainsInfo {
  std::vector<std::shared_ptr<LogicalChain>> ordered_logical_chains;
  std::vector<const OpNode*> ordered_acc_op_nodes;
  std::shared_ptr<LogicalChain> after_acc_logical_chain;
  const ParallelDesc* seed_parallel_desc;
  PlacementLogicalChainsInfo() : seed_parallel_desc(nullptr) {}
};

std::string GenParallelConfKey(const ParallelConf& conf) {
  std::string ret = conf.device_tag();
  for (const auto& name : conf.device_name()) { ret += ("-" + name); }
  return ret;
}

void InitPlacementLogicalChainsInfoFromSet(
    const std::shared_ptr<LogicalChain>& logical_chain,
    const HashSet<const OpNode*>& origin_logical_chain,
    const HashMap<const OpNode*, int64_t>& op_node2global_order,
    const std::function<bool(const OpNode*, const OpNode*)>& CmpOpNodeOrder) {
  auto* logical_chain_ordered_nodes = &logical_chain->ordered_op_nodes;
  CHECK(logical_chain_ordered_nodes->empty());
  logical_chain_ordered_nodes->assign(origin_logical_chain.begin(), origin_logical_chain.end());
  std::sort(logical_chain_ordered_nodes->begin(), logical_chain_ordered_nodes->end(),
            CmpOpNodeOrder);
  const OpNode* begin_op = logical_chain_ordered_nodes->front();
  const OpNode* end_op = logical_chain_ordered_nodes->back();
  int64_t begin_op_global_order = op_node2global_order.at(begin_op);
  int64_t end_op_global_order = op_node2global_order.at(end_op);
  CHECK(begin_op != end_op);
  CHECK_LT(begin_op_global_order, end_op_global_order);
}

void CreateAfterAccLogicalChain(const std::shared_ptr<LogicalChain>& after_acc_logical_chain,
                                const std::vector<const OpNode*>& ordered_acc_op_nodes,
                                const ParallelDesc& seed_parallel_desc) {
  // Meta time shape (1, 1)
  std::shared_ptr<const Shape> seed_time_shape = std::make_shared<const Shape>(Shape({1, 1}));
  HashSet<const OpNode*> visited;
  HashSet<const OpNode*> after_acc_chain_ops;
  std::queue<const OpNode*> queued_nodes;
  auto SearchToNextNode = [&](const OpNode* cur_node, const OpNode* next_node, const OpEdge* edge) {
    if (visited.find(next_node) == visited.end() && (!IsBreakpointOpNode(next_node))
        && next_node->parallel_desc().EqualsIgnoringHierarchy(seed_parallel_desc)
        && SharedPtrShapeEqual(GetOpNodeFastestTimeShape(next_node), seed_time_shape)
        && !NeedInsertBoxingBetweenOpNodes(cur_node, next_node, edge)) {
      CHECK(visited.insert(next_node).second);
      queued_nodes.push(next_node);
    }
  };

  for (const OpNode* acc_node : ordered_acc_op_nodes) {
    for (const OpEdge* out_edge : acc_node->out_edges()) {
      const OpNode* seed_node = out_edge->dst_node();
      SearchToNextNode(acc_node, seed_node, out_edge);
    }
  }

  while (!queued_nodes.empty()) {
    const OpNode* cur_node = queued_nodes.front();
    queued_nodes.pop();

    CHECK(after_acc_chain_ops.insert(cur_node).second);

    for (const OpEdge* in_edge : cur_node->in_edges()) {
      // NOTE(chengcheng): maybe bad case for too early source op before repeat.
      SearchToNextNode(cur_node, in_edge->src_node(), in_edge);
    }
    for (const OpEdge* out_edge : cur_node->out_edges()) {
      SearchToNextNode(cur_node, out_edge->dst_node(), out_edge);
    }
  }

  if (after_acc_chain_ops.size() > 1) {
    for (const OpNode* node : after_acc_chain_ops) {
      after_acc_logical_chain->ordered_op_nodes.push_back(node);
    }
    CHECK_EQ(after_acc_logical_chain->ordered_op_nodes.size(), after_acc_chain_ops.size());
  }
}

Maybe<void> LogicalChainPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  std::vector<const OpNode*> ordered_op_nodes;
  HashMap<const OpNode*, int64_t> op_node2global_order;
  HashMap<std::string, OperatorConf> mut_op_name2conf;
  // TODO(chengcheng) : better order for memory.
  std::shared_ptr<const Shape> seed_time_shape = std::make_shared<const Shape>(Shape({1, 1}));
  op_graph.TopoForEachNodeWithCtrlEdge([&](const OpNode* node) {
    ordered_op_nodes.emplace_back(node);
    op_node2global_order.emplace(node, ordered_op_nodes.size() - 1);
    std::shared_ptr<const Shape> this_time_shape = GetOpNodeFastestTimeShape(node);
    if (this_time_shape->elem_cnt() > seed_time_shape->elem_cnt()) {
      seed_time_shape = this_time_shape;
    }
    mut_op_name2conf.emplace(node->op().op_name(), node->op().op_conf());
  });

  VLOG(2) << " seed time shape = " << seed_time_shape->ToString();

  std::vector<HashSet<const OpNode*>> logical_chains;
  GetLogicalChainsWithTimeShape(&logical_chains, ordered_op_nodes, seed_time_shape);
  if (logical_chains.size() == 0) { return Maybe<void>::Ok(); }

  int64_t logical_chain_id = 0;
  auto NewLogicalChainId = [&]() { return logical_chain_id++; };

  auto CmpOpNodeOrder = [&](const OpNode* lhs, const OpNode* rhs) {
    return op_node2global_order.at(lhs) < op_node2global_order.at(rhs);
  };
  auto CmpLogicalChainOrder = [&](const std::shared_ptr<LogicalChain>& lhs,
                                  const std::shared_ptr<LogicalChain>& rhs) {
    int64_t lhs_begin_op_global_order = op_node2global_order.at(lhs->ordered_op_nodes.front());
    int64_t rhs_begin_op_global_order = op_node2global_order.at(rhs->ordered_op_nodes.front());
    return lhs_begin_op_global_order < rhs_begin_op_global_order;
  };
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();

  HashMap<std::string, PlacementLogicalChainsInfo> placement2logical_chains;
  for (const auto& origin_logical_chain : logical_chains) {
    const OpNode* rand_node = *origin_logical_chain.begin();
    const ParallelDesc& this_parallel_desc = rand_node->parallel_desc();
    std::string key = GenParallelConfKey(this_parallel_desc.parallel_conf());
    auto it = placement2logical_chains.find(key);
    if (it == placement2logical_chains.end()) {
      it = placement2logical_chains.emplace(key, PlacementLogicalChainsInfo()).first;
      it->second.seed_parallel_desc = &this_parallel_desc;
    }
    auto& info = it->second;
    info.ordered_logical_chains.emplace_back(std::make_shared<LogicalChain>());
    InitPlacementLogicalChainsInfoFromSet(info.ordered_logical_chains.back(), origin_logical_chain,
                                          op_node2global_order, CmpOpNodeOrder);
  }

  for (auto& pair : placement2logical_chains) {
    std::sort(pair.second.ordered_logical_chains.begin(), pair.second.ordered_logical_chains.end(),
              CmpLogicalChainOrder);
  }

  for (const OpNode* this_node : ordered_op_nodes) {
    if (IsAccOpNode(this_node)) {
      const ParallelDesc& this_parallel_desc = this_node->parallel_desc();
      std::string key = GenParallelConfKey(this_parallel_desc.parallel_conf());
      auto it = placement2logical_chains.find(key);
      if (it != placement2logical_chains.end()) {
        it->second.ordered_acc_op_nodes.emplace_back(this_node);
      }
    }
    JUST(MapAt(mut_op_name2conf, this_node->op().op_name()))
        .set_logical_order(JUST(MapAt(op_node2global_order, this_node)));
  }

  auto InsertCtrlEdgeInChain = [&](const std::vector<const OpNode*>& ordered_op_nodes) {
    for (int64_t i = 1; i < ordered_op_nodes.size(); ++i) {
      const OpNode* this_node = CHECK_JUST(VectorAt(ordered_op_nodes, i));
      const OpNode* prev_node = CHECK_JUST(VectorAt(ordered_op_nodes, i - 1));
      const std::string& this_op_name = this_node->op().op_name();
      const std::string& prev_op_name = prev_node->op().op_name();
      if (!IsReachable(prev_op_name, this_op_name)) {
        CHECK_JUST(MapAt(mut_op_name2conf, this_op_name)).add_ctrl_in_op_name(prev_op_name);
      }
    }
  };

  auto InsertLogicalChainId = [&](const std::vector<const OpNode*>& ordered_op_nodes,
                                  const int64_t logical_chain_id) {
    for (const OpNode* op_node : ordered_op_nodes) {
      CHECK_JUST(MapAt(mut_op_name2conf, op_node->op().op_name()))
          .set_logical_chain_id(logical_chain_id);
    }
  };

  for (auto& pair : placement2logical_chains) {
    const auto& placement = pair.first;
    auto& info = pair.second;
    CHECK_GE(info.ordered_logical_chains.size(), 1);
    for (auto& logical_chain : info.ordered_logical_chains) {
      logical_chain->logical_chain_id = NewLogicalChainId();
      InsertLogicalChainId(logical_chain->ordered_op_nodes, logical_chain->logical_chain_id);
      InsertCtrlEdgeInChain(logical_chain->ordered_op_nodes);
    }

    for (const auto& logical_chain : info.ordered_logical_chains) {
      VLOG(3) << " In placement: " << placement
              << " logical_chain_id: " << logical_chain->logical_chain_id
              << " has op num = " << logical_chain->ordered_op_nodes.size();

      for (int i = 0; i < logical_chain->ordered_op_nodes.size(); ++i) {
        const OpNode* ordered_op = JUST(VectorAt(logical_chain->ordered_op_nodes, i));
        VLOG(3) << " ChainId: " << logical_chain_id << " order: " << i
                << " op_name: " << ordered_op->op().op_name()
                << " global_order: " << JUST(MapAt(op_node2global_order, ordered_op));
      }
    }

    // NOTE(chengcheng): create logical chain after acc, and merge with first logical chain.
    const std::vector<const OpNode*>& ordered_acc_op_nodes = info.ordered_acc_op_nodes;
    if (!ordered_acc_op_nodes.empty()) {
      info.after_acc_logical_chain = std::make_shared<LogicalChain>();
      CreateAfterAccLogicalChain(info.after_acc_logical_chain, ordered_acc_op_nodes,
                                 *info.seed_parallel_desc);
      auto& acc_chain_order_ops = info.after_acc_logical_chain->ordered_op_nodes;
      if (acc_chain_order_ops.size() > 1) {
        info.after_acc_logical_chain->logical_chain_id = NewLogicalChainId();
        std::sort(acc_chain_order_ops.begin(), acc_chain_order_ops.end(), CmpOpNodeOrder);
        const auto& first_chain = info.ordered_logical_chains.front();
        const OpNode* first_chain_src_op = first_chain->ordered_op_nodes.front();
        const OpNode* first_chain_sink_op = first_chain->ordered_op_nodes.back();

        const OpNode* acc_chain_src_op = acc_chain_order_ops.front();
        const OpNode* acc_chain_sink_op = acc_chain_order_ops.back();

        // NOTE(chengcheng): find last op can insert acc ctrl tick.
        while (
            (!acc_chain_sink_op->op().op_conf().has_user_conf())
            || IsReachable(acc_chain_sink_op->op().op_name(), first_chain_src_op->op().op_name())) {
          VLOG(3) << " cannot insert acc ctrl edge between: [" << first_chain_src_op->op().op_name()
                  << "] -> [" << acc_chain_sink_op->op().op_name() << "] , debug info :\n"
                  << first_chain_src_op->op().op_conf().DebugString() << "\n"
                  << acc_chain_sink_op->op().op_conf().DebugString() << "\n";

          VLOG(3) << "remove op : " << acc_chain_sink_op->op().op_name()
                  << " from after acc logical chain: "
                  << info.after_acc_logical_chain->logical_chain_id;
          acc_chain_order_ops.pop_back();
          if (acc_chain_order_ops.size() > 1) {
            acc_chain_sink_op = acc_chain_order_ops.back();
          } else {
            acc_chain_sink_op = nullptr;
            break;
          }
        }
        if (acc_chain_sink_op == nullptr) { continue; }

        // NOTE(chengcheng): find first op can insert acc tick.
        while (IsReachable(acc_chain_src_op->op().op_name(), first_chain_sink_op->op().op_name())) {
          VLOG(3) << " cannot insert acc tick edge between: ["
                  << first_chain_sink_op->op().op_name() << "] -> ["
                  << acc_chain_src_op->op().op_name() << "] , debug info :\n"
                  << first_chain_sink_op->op().op_conf().DebugString() << "\n"
                  << acc_chain_src_op->op().op_conf().DebugString() << "\n";

          VLOG(3) << "remove op : " << acc_chain_src_op->op().op_name()
                  << " from after acc logical chain: "
                  << info.after_acc_logical_chain->logical_chain_id;
          acc_chain_order_ops.erase(acc_chain_order_ops.begin());
          if (acc_chain_order_ops.size() > 1) {
            acc_chain_src_op = acc_chain_order_ops.front();
          } else {
            acc_chain_src_op = nullptr;
            break;
          }
        }
        if (acc_chain_src_op == nullptr) { continue; }

        // NOTE(chengcheng):
        //   1.add acc ctrl tick between first chain src to acc chain sink for memory lock.
        const int64_t acc_num = job_builder->job().job_conf().num_gradient_accumulation_steps();
        CHECK_GT(acc_num, 1);
        const auto& fc_src_obns = first_chain_src_op->op().output_bns();
        CHECK(!fc_src_obns.empty());
        const std::string& first_chain_src_out_lbn =
            GenLogicalBlobName(first_chain_src_op->op().BnInOp2Lbi(fc_src_obns.Get(0)));

        VLOG(3) << " first_chain_src_out_lbn : " << first_chain_src_out_lbn;
        user_op::UserOpConfWrapper acc_ctrl_tick_op =
            user_op::UserOpConfWrapperBuilder("Sys-AccCtrlTick4MergeFirstAccChain-" + NewUniqueId())
                .OpTypeName("acc_ctrl_tick")
                .Input("in", first_chain_src_out_lbn)
                .Output("out")
                .ScopeSymbolId(first_chain_src_op->op().op_conf().scope_symbol_id())
                .Attr<int32_t>("max_acc_num", acc_num)
                .Build();

        OperatorConf& acc_chain_sink_op_conf =
            JUST(MapAt(mut_op_name2conf, acc_chain_sink_op->op().op_name()));
        CHECK(acc_chain_sink_op_conf.has_user_conf());
        (*acc_chain_sink_op_conf.mutable_user_conf()
              ->mutable_input())[user_op::kUserSourceOpTickInputArgName]
            .add_s(acc_ctrl_tick_op.output("out", 0));
        JUST(job_builder->AddOp(first_chain_src_op->parallel_desc().parallel_conf(),
                                acc_ctrl_tick_op.op_conf()));
        VLOG(3) << " Insert acc ctrl tick between: [" << first_chain_src_op->op().op_name()
                << "] -> [" << acc_chain_sink_op->op().op_name() << "]";

        // NOTE(chengcheng):
        //   2.add acc tick between first chain sink to acc chain src for strict exec order.
        const auto& fc_sink_obns = first_chain_sink_op->op().output_bns();
        CHECK(!fc_sink_obns.empty());
        const std::string first_chain_sink_lbn =
            GenLogicalBlobName(first_chain_sink_op->op().BnInOp2Lbi(fc_sink_obns.Get(0)));
        VLOG(3) << " first_chain_sink_lbn : " << first_chain_sink_lbn;

        user_op::UserOpConfWrapper cast_to_tick_op =
            user_op::UserOpConfWrapperBuilder("Sys-LogicalChainSink-CastToTick-" + NewUniqueId())
                .OpTypeName("cast_to_tick")
                .Input("in", first_chain_sink_lbn)
                .Output("out")
                .ScopeSymbolId(first_chain_sink_op->op().op_conf().scope_symbol_id())
                .Build();

        CHECK_JUST(job_builder->AddOp(first_chain_sink_op->parallel_desc().parallel_conf(),
                                      cast_to_tick_op.op_conf()));

        std::string acc_tick_output_lbn = cast_to_tick_op.output("out", 0);
        if (!IsAccOpNode(first_chain_sink_op)) {
          // NOTE(chengcheng): Acc Op can be merged in fw/bw chain, if the last op is acc op,
          //  there is no need and CANNOT insert acc tick op.

          OperatorConf sink_acc_tick_conf;
          sink_acc_tick_conf.set_name(std::string("Sys-LogicalChainSink-AccTick_") + NewUniqueId());
          sink_acc_tick_conf.set_scope_symbol_id(
              first_chain_sink_op->op().op_conf().scope_symbol_id());
          auto* acc_conf = sink_acc_tick_conf.mutable_acc_tick_conf();
          acc_conf->set_one(cast_to_tick_op.output("out", 0));
          acc_conf->set_acc("acc");
          acc_conf->set_max_acc_num(acc_num);
          acc_tick_output_lbn = GenLogicalBlobName(sink_acc_tick_conf.name(), "acc");

          VLOG(3) << " insert acc tick op : " << sink_acc_tick_conf.name()
                  << " of last op in fw/bw chain.";

          CHECK_JUST(job_builder->AddOp(first_chain_sink_op->parallel_desc().parallel_conf(),
                                        sink_acc_tick_conf));
        }

        OperatorConf sink_final_tick_conf;
        sink_final_tick_conf.set_name(std::string("Sys-LogicalChainSink-FinalTick-DeviceTick_")
                                      + NewUniqueId());
        sink_final_tick_conf.set_scope_symbol_id(
            first_chain_sink_op->op().op_conf().scope_symbol_id());
        auto* tick_conf = sink_final_tick_conf.mutable_device_tick_conf();
        tick_conf->add_tick(acc_tick_output_lbn);
        tick_conf->set_out("out");

        CHECK_JUST(job_builder->AddOp(first_chain_sink_op->parallel_desc().parallel_conf(),
                                      sink_final_tick_conf));

        JUST(MapAt(mut_op_name2conf, acc_chain_src_op->op().op_name()))
            .add_ctrl_in_op_name(sink_final_tick_conf.name());

        VLOG(3) << " Insert acc tick between: [" << first_chain_sink_op->op().op_name() << "] -> ["
                << acc_chain_src_op->op().op_name() << "]";

        // NOTE(chengcheng):
        //   3. merge first chain and acc chain
        MergedLogicalChainIdGroup* group = job_builder->add_logical_chain_groups();
        group->add_logical_chain_id_list(first_chain->logical_chain_id);
        group->add_logical_chain_id_list(info.after_acc_logical_chain->logical_chain_id);
        VLOG(3) << " Merge acc chain : " << info.after_acc_logical_chain->logical_chain_id
                << " to first logcal chain : " << first_chain->logical_chain_id;
      }

      VLOG(3) << " In placement: " << placement
              << " AccLogicalChain: " << info.after_acc_logical_chain->logical_chain_id
              << " has op num = " << acc_chain_order_ops.size();

      for (int i = 0; i < acc_chain_order_ops.size(); ++i) {
        const OpNode* ordered_op = JUST(VectorAt(acc_chain_order_ops, i));
        VLOG(3) << " AfterAccChainId: " << info.after_acc_logical_chain->logical_chain_id
                << " order: " << i << " op_name: " << ordered_op->op().op_name()
                << " global_order: " << JUST(MapAt(op_node2global_order, ordered_op));
      }

      InsertLogicalChainId(acc_chain_order_ops, info.after_acc_logical_chain->logical_chain_id);
      InsertCtrlEdgeInChain(acc_chain_order_ops);
    }
  }

  // NOTE(chengcheng): update global order and chain id for ops.
  for (const auto& pair : mut_op_name2conf) { JUST(job_builder->MutOpOnlyOnce(pair.second)); }

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("LogicalChainPass", LogicalChainPass);

}  // namespace oneflow
