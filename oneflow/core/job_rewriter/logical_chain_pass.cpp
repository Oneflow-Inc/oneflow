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

namespace oneflow {

DEFINE_ENV_BOOL(ENABLE_LOGICAL_CHAIN, true);

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

  bool IsEnabled(const JobPassCtx& ctx) const {
    return EnvBool<ENABLE_LOGICAL_CHAIN>();
  }

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
        || op_conf.has_acc_tick_conf()  
        || op_conf.has_critical_section_wait_tick_conf()
        || op_conf.has_critical_section_callback_tick_conf() /* tick */
        || op_conf.has_input_conf()
        || op_conf.has_output_conf() /* io */
        || op_conf.has_wait_and_send_ids_conf()
        || op_conf.has_callback_notify_conf() /* ctrl */
        || op_conf.has_image_decoder_random_crop_resize_conf() /* gpu decode */) {
      return true;
  }

  if (op_conf.has_user_conf()) {
    const std::string& user_type_name = op_conf.user_conf().op_type_name();
    // TODO(chengcheng): acc node can be merged in chain.
    if (user_type_name == "repeat" || user_type_name == "acc" || user_type_name == "pack"
        || user_type_name == "unpack" || user_type_name == "identity_buffer"
        || user_type_name == "copy_h2d" || user_type_name == "copy_d2h") {
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

void GetChainsWithTimeShape(std::vector<HashSet<const OpNode*>>* ret,
                                             const OpGraph& op_graph,
                                             const std::vector<const OpNode*>& order, 
                                             const std::shared_ptr<const Shape>& seed_time_shape) {
  HashSet<const OpNode*> visited;
  for (const OpNode* seed_node : order) {
    if (visited.find(seed_node) != visited.end()) { continue; }
    CHECK(visited.insert(seed_node).second);
    const ParallelDesc& seed_parallel_desc = seed_node->parallel_desc();
    // TODO(chengcheng): support cpu chain.
    if (seed_parallel_desc.device_type() == DeviceType::kCPU) { continue; }
    if (!SharedPtrShapeEqual(GetOpNodeFastestTimeShape(seed_node), seed_time_shape) { continue; }
    if (IsBreakpointOpNode(seed_node)) { continue; }

    HashSet<const OpNode*> this_subgraph;
    std::queue<const OpNode*> queued_nodes;

    queued_nodes.push(seed_node);
    while (!queued_nodes.empty()) {
      const OpNode* cur_node = queued_nodes.front();
      queued_nodes.pop();

      CHECK(cur_node->parallel_desc().EqualsIgnoringHierarchy(seed_parallel_desc));
      CHECK(this_subgraph.insert(cur_node).second);

      cur_node->ForEachNodeOnInOutEdge([&](const OpNode* next_node) {
        if (visited.find(next_node) == visited.end() && (!IsBreakpointOpNode(next_node))
            && next_node->parallel_desc().EqualsIgnoringHierarchy(seed_parallel_desc)
            && SharedPtrShapeEqual(GetOpNodeFastestTimeShape(next_node), seed_time_shape)) {
          CHECK(visited.insert(next_node).second);
          queued_nodes.push(next_node);
        }
      });
    }

    if (this_subgraph.size() > 1) {
      ret->emplace_back(HashSet<const OpNode*>());
      ret->back().swap(this_subgraph);
    }
  }

  std::sort(ret->begin(), ret->end(),
            [](const HashSet<const OpNode*>& lhs, const HashSet<const OpNode*>& rhs) {
              return lhs.size() > rhs.size();
            });
}

struct LogicalChain {
  int64_t logical_chain_id;
  std::vector<const OpNode*> ordered_op_nodes;
  int64_t begin_op_global_order;
  int64_t end_op_global_order;
  const OpNode* begin_op;
  const OpNode* end_op;
};

struct PlacementLogicalChainsInfo {
  std::vector<std::shared_ptr<LogicalChain>> ordered_logical_chains;
  std::vector<const OpNode*> ordered_acc_op_nodes;
  const ParallelDesc* seed_parallel_desc;
};

std::string GenParallelConfKey(const ParallelConf& conf) {
  std::string ret = conf.device_tag();
  for (const auto& name : conf.device_name()) { ret += ("-" + name); }
  return ret;
}

Maybe<void> LogicalChainPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  std::vector<const OpNode*> ordered_op_nodes;
  HashMap<const OpNode*, int64_t> op_node2global_order;
  // TODO(chengcheng) : better order for memory.
  std::shared_ptr<const Shape> seed_time_shape = std::make_shared<const Shape>(Shape({1, 1}));
  op_graph.TopoForEachNodeWithCtrlEdge([&](const OpNode* node) {
    ordered_op_nodes.emplace_back(node);
    op_node2global_order.emplace(node, ordered_op_nodes.size() - 1);
    std::shared_ptr<const Shape> this_time_shape = GetOpNodeFastestTimeShape(node);
    if (this_time_shape->elem_cnt() > seed_time_shape->elem_cnt()) {
      seed_time_shape = this_time_shape;
    }
  });

  VLOG(2) << " seed time shape = " << seed_time_shape->ToString();

  std::vector<HashSet<const OpNode*>> logical_chains;
  GetChainsWithTimeShape(&logical_chains, op_graph, ordered_op_nodes, seed_time_shape);
  if (logical_chains.size() == 0) { return Maybe<void>::Ok(); }

  int64_t logical_chain_id = 0;
  auto NewLogicalChainId = [&]() { return logical_chain_id++};

  auto CmpOpNodeOrder = [&](const OpNode* lhs, const OpNode* rhs) {
    return op_node2global_order.at(lhs) < op_node2global_order.at(rhs);
  };
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();

  HashMap<std::string, PlacementLogicalChainsInfo> placement2logical_chains;
  for (const auto& origin_logical_chain : logical_chains) {
    const OpNode* rand_node = *subgraph.begin();
    const ParallelDesc& this_parallel_desc = rand_node->parallel_desc();
    std::string key = GenParallelConfKey(this_parallel_desc.parallel_conf());
    const std::shared_ptr<const Shape>& this_time_shape = GetOpNodeFastestTimeShape(rand_node);
    auto it = placement2subgraphs.find(key);
    if (it == placement2subgraphs.end()) {
      it = placement2subgraphs.emplace(key, PlacementNcclSubGraghsInfo()).first;
      auto& info = it->second;
      info.seed_parallel_desc = &this_parallel_desc;
      info.seed_time_shape = this_time_shape;
      info.ordered_subgraph.emplace_back(std::make_shared<InsertNcclSubGraph>());
      InitInsertNcclSubGraphInfoFromSet(info.ordered_subgraph.back(), subgraph,
                                        op_node2global_order, CmpOpNodeOrder);
    } else {
      auto& info = it->second;
      if (SharedPtrShapeEqual(info.seed_time_shape, this_time_shape)) {
        CHECK(this_parallel_desc.EqualsIgnoringHierarchy(*info.seed_parallel_desc));
        std::shared_ptr<InsertNcclSubGraph> nccl_subgraph_info =
            std::make_shared<InsertNcclSubGraph>();
        InitInsertNcclSubGraphInfoFromSet(nccl_subgraph_info, subgraph, op_node2global_order,
                                          CmpOpNodeOrder);
        CHECK_GT(info.ordered_subgraph.size(), 0);
        const auto& first_graph = info.ordered_subgraph.front();
        const auto& last_graph = info.ordered_subgraph.back();
        int64_t first_order = first_graph->begin_op_global_order;
        int64_t last_order = last_graph->end_op_global_order;
        if (nccl_subgraph_info->end_op_global_order < first_order) {
          if (IsReachable(nccl_subgraph_info->end_op->op().op_name(),
                          first_graph->begin_op->op().op_name())) {
            info.ordered_subgraph.insert(info.ordered_subgraph.begin(), nccl_subgraph_info);
          }
        } else if (nccl_subgraph_info->begin_op_global_order > last_order) {
          if (IsReachable(last_graph->end_op->op().op_name(),
                          nccl_subgraph_info->begin_op->op().op_name())) {
            info.ordered_subgraph.emplace_back(nccl_subgraph_info);
          }
        } else {
          auto before = info.ordered_subgraph.begin();
          auto next = before + 1;
          while (next != info.ordered_subgraph.end()) {
            if ((*before)->end_op_global_order < nccl_subgraph_info->begin_op_global_order
                && nccl_subgraph_info->end_op_global_order < (*next)->begin_op_global_order) {
              if (IsReachable((*before)->end_op->op().op_name(),
                              nccl_subgraph_info->begin_op->op().op_name())
                  && IsReachable(nccl_subgraph_info->end_op->op().op_name(),
                                 (*next)->begin_op->op().op_name())) {
                info.ordered_subgraph.insert(next, nccl_subgraph_info);
              }
              break;
            }
            before = next;
            next++;
          }
        }
      }
    }
  }

  for (const OpNode* this_node : ordered_op_nodes) {
    if (IsAccOpNode(this_node)) {
      const ParallelDesc& this_parallel_desc = this_node->parallel_desc();
      std::string key = GenParallelConfKey(this_parallel_desc.parallel_conf());
      auto it = placement2subgraphs.find(key);
      if (it != placement2subgraphs.end()) {
        it->second.ordered_acc_op_nodes.emplace_back(this_node);
      }
    }
  }

  for (auto& pair : placement2subgraphs) {
    PlacementNcclSubGraghsInfo& info = pair.second;
    for (int i = 0; i < info.ordered_subgraph.size() - 1; i++) {
      CHECK_LT(info.ordered_subgraph.at(i)->end_op_global_order,
               info.ordered_subgraph.at(i + 1)->begin_op_global_order);
    }

    // NOTE(chengcheng): insert nccl ops for each subgraph
    uint32_t stream_offset = 0;
    int64_t total_op_num = 0;
    for (int i = 0; i < info.ordered_subgraph.size(); i++) {
      auto& ordered_op_nodes = info.ordered_subgraph.at(i)->ordered_op_nodes;
      InsertNcclLogicalOpsInSubGraph(op_graph, job_builder, ordered_op_nodes, IsReachable, i,
                                     &stream_offset);
      total_op_num += ordered_op_nodes.size();
    }
    if (stream_offset >= 2 && total_op_num >= 1000) {
      LOG(WARNING) << " In Graph: " << job_builder->job().job_conf().job_name()
                   << " Placement: " << pair.first << " the total_op_num = " << total_op_num
                   << " and has " << stream_offset
                   << " different nccl stream which is possible to trigger cuda stream kernel "
                      "launch upper limit."
                   << " So the nccl logical kernel will from async to sync exec, which may affect "
                      "performance.";
      EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
      comm_mgr->SetAsyncLaunchNcclLogicalKernel(false);
    }

    // NOTE(chengcheng): insert acc for all subgraph with same placement group
    const OpNode* bw_sink_op = info.ordered_subgraph.back()->end_op;
    const std::vector<const OpNode*>& ordered_acc_op_nodes = info.ordered_acc_op_nodes;

    if (!ordered_acc_op_nodes.empty()) {
      InsertBwSinkAccTickAndNcclLogicalOpsInPlacementGroupAfterAcc(
          op_graph, job_builder, ordered_acc_op_nodes, op_node2global_order, bw_sink_op);
    }
  }

  return Maybe<void>::Ok();
}




bool IsOpEdgeAllowInsertNccl(const OpEdge* edge,
                             const std::shared_ptr<const Shape>& seed_time_shape) {
  const OpNode* src_node = edge->src_node();
  const OpNode* dst_node = edge->dst_node();
  const ParallelDesc& src_parallel_desc = src_node->parallel_desc();
  return src_parallel_desc.device_type() == DeviceType::kCUDA
         && src_parallel_desc.parallel_num() > 1
         && src_parallel_desc.EqualsIgnoringHierarchy(dst_node->parallel_desc())
         && SharedPtrShapeEqual(GetOpNodeFastestTimeShape(src_node), seed_time_shape)
         && SharedPtrShapeEqual(GetOpNodeFastestTimeShape(dst_node), seed_time_shape);
}

struct InsertedNcclInfo {
  OperatorConf nccl_op_conf;
  ParallelConf nccl_parallel_conf;
  int64_t order;
  const OpNode* src_node;
  const OpNode* dst_node;
  std::string debug_str;
};

void InsertNcclLogicalOpsAfterAcc(const OpGraph& op_graph,
                                  const HashMap<const OpNode*, int64_t>& op_node2global_order,
                                  const std::vector<const OpNode*>& ordered_acc_op_nodes,
                                  const std::string& bw_sink_tick_op_name,
                                  HashMap<std::string, OperatorConf>* mut_consumer_name2op,
                                  std::vector<OperatorConf>* nccl_op_confs,
                                  std::vector<ParallelConf>* nccl_op_parallel_confs) {
  HashSet<const OpEdge*> visited;
  std::shared_ptr<const Shape> seed_time_shape = GetOpNodeFastestTimeShape(ordered_acc_op_nodes.front());
  std::vector<InsertedNcclInfo> nccl_op_infos;

  std::vector<const OpNode*> ordered_after_acc_subgraph;
  // NOTE(chengcheng): bfs for op_edge may create duplicated node.
  HashSet<const OpNode*> after_acc_subgraph_nodes;
  HashMap<const OpNode*, int64_t> op2subgraph_order;

  for (const OpNode* acc : ordered_acc_op_nodes) {
    std::queue<const OpEdge*> queued_edges;
    for (const OpEdge* op_edge : acc->out_edges()) {
      if (visited.find(op_edge) == visited.end()
          && IsOpEdgeAllowInsertNccl(op_edge, seed_time_shape)) {
        queued_edges.push(op_edge);
        CHECK(visited.insert(op_edge).second);
        if (!IsAccOpNode(op_edge->dst_node())) {
          after_acc_subgraph_nodes.insert(op_edge->dst_node());
        }
      }
    }

    auto NextEdgeNode2AfterAccSubGraph = [&](const OpEdge* next_edge, const OpNode* next_node) {
      if (visited.find(next_edge) == visited.end()
          && IsOpEdgeAllowInsertNccl(next_edge, seed_time_shape)) {
        CHECK(visited.insert(next_edge).second);
        queued_edges.push(next_edge);
        if (!IsAccOpNode(next_node)) { after_acc_subgraph_nodes.insert(next_node); }
      }
    };

    // bfs search each edge after acc allow insert nccl. try insert.
    while (!queued_edges.empty()) {
      const OpEdge* op_edge = queued_edges.front();
      queued_edges.pop();

      for (const LogicalBlobId& lbi : op_edge->lbis()) {
        const OpNode* src_node = op_edge->src_node();
        const OpNode* dst_node = op_edge->dst_node();
        const std::string& src_op_name = src_node->op().op_name();
        const std::string& dst_op_name = dst_node->op().op_name();
        OperatorConf nccl_op;
        ParallelDesc src_reduced_parallel_desc = op_edge->src_node()->parallel_desc();
        ParallelDesc dst_reduced_parallel_desc = op_edge->dst_node()->parallel_desc();
        NdSbp src_reduced_nd_sbp;
        NdSbp dst_reduced_nd_sbp;
        if (!TryBuildNcclLogicalOpConf(&nccl_op, op_edge->src_node(), op_edge->dst_node(), lbi,
                                       &src_reduced_parallel_desc, &dst_reduced_parallel_desc,
                                       &src_reduced_nd_sbp, &dst_reduced_nd_sbp)) {
          continue;
        }
        auto it = mut_consumer_name2op->find(dst_op_name);
        if (it == mut_consumer_name2op->end()) {
          auto ret_pair = mut_consumer_name2op->emplace(dst_op_name, dst_node->op().op_conf());
          CHECK(ret_pair.second);
          it = ret_pair.first;
        }
        // insert nccl op
        user_op::UserOpConfWrapper nccl_op_wrapper(nccl_op);
        for (const std::string& ibn : op_edge->lbi2ibns().at(lbi)) {
          std::string old_lbn = ReplaceInputLbnInOpCustomizedConf(&(it->second), ibn,
                                                                  nccl_op_wrapper.output("out", 0));
        }

        InsertedNcclInfo nccl_op_info;
        nccl_op_info.nccl_op_conf = nccl_op;
        nccl_op_info.nccl_parallel_conf = src_reduced_parallel_desc.parallel_conf();
        nccl_op_info.order = op_node2global_order.at(src_node);
        nccl_op_info.src_node = src_node;
        nccl_op_info.dst_node = dst_node;
        nccl_op_info.debug_str =
            (" After ACC insert nccl op: " + nccl_op.name() + " from [" + src_op_name
             + ", sbp=" + NdSbpToString(src_node->NdSbp4Lbi(lbi)) + "] to [" + dst_op_name
             + ", sbp=" + NdSbpToString(dst_node->NdSbp4Lbi(lbi))
             + ", src_order=" + std::to_string(nccl_op_info.order) + "]\n");
        nccl_op_infos.emplace_back(nccl_op_info);
      }

      // NOTE(chengcheng): BFS for all edges and nodes after acc.
      for (const OpEdge* dst_node_out_edge : op_edge->dst_node()->out_edges()) {
        NextEdgeNode2AfterAccSubGraph(dst_node_out_edge, dst_node_out_edge->dst_node());
      }
      for (const OpEdge* dst_node_in_edge : op_edge->dst_node()->in_edges()) {
        NextEdgeNode2AfterAccSubGraph(dst_node_in_edge, dst_node_in_edge->src_node());
      }
      for (const OpEdge* src_node_out_edge : op_edge->src_node()->out_edges()) {
        NextEdgeNode2AfterAccSubGraph(src_node_out_edge, src_node_out_edge->dst_node());
      }
      for (const OpEdge* src_node_in_edge : op_edge->src_node()->in_edges()) {
        NextEdgeNode2AfterAccSubGraph(src_node_in_edge, src_node_in_edge->src_node());
      }
    }
  }

  for (const auto* node : after_acc_subgraph_nodes) { ordered_after_acc_subgraph.push_back(node); }

  CHECK_EQ(after_acc_subgraph_nodes.size(), ordered_after_acc_subgraph.size());

  std::sort(nccl_op_infos.begin(), nccl_op_infos.end(),
            [](const InsertedNcclInfo& lhs, const InsertedNcclInfo& rhs) {
              return lhs.order < rhs.order;
            });

  std::sort(ordered_after_acc_subgraph.begin(), ordered_after_acc_subgraph.end(),
            [&](const OpNode* lhs, const OpNode* rhs) {
              return op_node2global_order.at(lhs) < op_node2global_order.at(rhs);
            });

  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();

  for (int64_t i = 0; i < ordered_after_acc_subgraph.size(); ++i) {
    op2subgraph_order.emplace(ordered_after_acc_subgraph.at(i), i);
  }

  for (int64_t i = 1; i < ordered_after_acc_subgraph.size(); ++i) {
    const OpNode* this_node = ordered_after_acc_subgraph.at(i);
    const OpNode* pre_node = ordered_after_acc_subgraph.at(i - 1);
    const std::string& this_op_name = this_node->op().op_name();
    const std::string& pre_op_name = pre_node->op().op_name();
    // build ctrl edge if need.
    if (!IsReachable(pre_op_name, this_op_name)) {
      auto it = mut_consumer_name2op->find(this_op_name);
      if (it == mut_consumer_name2op->end()) {
        auto ret_pair = mut_consumer_name2op->emplace(this_op_name, this_node->op().op_conf());
        CHECK(ret_pair.second);
        it = ret_pair.first;
      }
      OperatorConf* mut_op_conf = &(it->second);
      mut_op_conf->add_ctrl_in_op_name(pre_op_name);
    }
  }

  for (int64_t i = 0; i < nccl_op_infos.size(); ++i) {
    auto& info = nccl_op_infos.at(i);
    if (i == 0) {
      info.nccl_op_conf.add_ctrl_in_op_name(bw_sink_tick_op_name);
    } else {
      info.nccl_op_conf.add_ctrl_in_op_name(nccl_op_infos.at(i - 1).nccl_op_conf.name());
    }

    nccl_op_confs->emplace_back(info.nccl_op_conf);
    nccl_op_parallel_confs->emplace_back(info.nccl_parallel_conf);
    VLOG(3) << info.debug_str;

    // NOTE(chengcheng): Try add ctrl between nccl and src op next node for strict exec order.
    auto src_op_it = op2subgraph_order.find(info.src_node);
    if (src_op_it != op2subgraph_order.end()) {
      const int64_t src_sub_order = src_op_it->second;
      const int64_t next_sub_order = src_sub_order + 1;
      if (next_sub_order < ordered_after_acc_subgraph.size()) {
        const OpNode* next_op = ordered_after_acc_subgraph.at(next_sub_order);
        const std::string& next_op_name = next_op->op().op_name();
        const std::string& dst_op_name = info.dst_node->op().op_name();
        if (next_op_name != dst_op_name) {
          if (mut_consumer_name2op->find(next_op_name) == mut_consumer_name2op->end()) {
            CHECK(mut_consumer_name2op->emplace(next_op_name, next_op->op().op_conf()).second);
          }
          // NOTE(chengcheng): MUST add ctrl edge for strict exec orde
          mut_consumer_name2op->at(next_op_name).add_ctrl_in_op_name(info.nccl_op_conf.name());
        }
      }
    }
  }
}

struct InsertNcclSubGraph {
  std::vector<const OpNode*> ordered_op_nodes;
  int64_t begin_op_global_order;
  int64_t end_op_global_order;
  const OpNode* begin_op;
  const OpNode* end_op;
};

struct PlacementNcclSubGraghsInfo {
  std::vector<std::shared_ptr<InsertNcclSubGraph>> ordered_subgraph;
  std::vector<const OpNode*> ordered_acc_op_nodes;
  const ParallelDesc* seed_parallel_desc;
  std::shared_ptr<const Shape> seed_time_shape;
};

void InitInsertNcclSubGraphInfoFromSet(
    std::shared_ptr<InsertNcclSubGraph> nccl_subgraph_info, const HashSet<const OpNode*>& subgraph,
    const HashMap<const OpNode*, int64_t>& op_node2global_order,
    const std::function<bool(const OpNode*, const OpNode*)>& CmpOpNodeOrder) {
  auto* subgraph_ordered_nodes = &nccl_subgraph_info->ordered_op_nodes;
  subgraph_ordered_nodes->assign(subgraph.begin(), subgraph.end());
  std::sort(subgraph_ordered_nodes->begin(), subgraph_ordered_nodes->end(), CmpOpNodeOrder);
  nccl_subgraph_info->begin_op = subgraph_ordered_nodes->front();
  nccl_subgraph_info->end_op = subgraph_ordered_nodes->back();
  nccl_subgraph_info->begin_op_global_order = op_node2global_order.at(nccl_subgraph_info->begin_op);
  nccl_subgraph_info->end_op_global_order = op_node2global_order.at(nccl_subgraph_info->end_op);
  CHECK(nccl_subgraph_info->begin_op != nccl_subgraph_info->end_op);
  CHECK_LT(nccl_subgraph_info->begin_op_global_order, nccl_subgraph_info->end_op_global_order);
}

constexpr uint32_t kMaxNcclComputeStreamCount = 8;

std::string GetStreamIndexName(uint32_t id) { return "NCCL_COMPUTE_" + std::to_string(id); }

void InsertNcclLogicalOpsInSubGraph(
    const OpGraph& op_graph, JobBuilder* job_builder,
    const std::vector<const OpNode*>& subgraph_order,
    const std::function<bool(const std::string&, const std::string&)>& IsReachable,
    const int32_t subgraph_id_in_same_placement_group, uint32_t* stream_offset) {
  HashMap<const OpNode*, int64_t> node2subgraph_order;
  node2subgraph_order.reserve(subgraph_order.size());
  for (int64_t i = 0; i < subgraph_order.size(); ++i) {
    CHECK(node2subgraph_order.emplace(subgraph_order.at(i), i).second);
  }

  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    VLOG(3) << " Try insert nccl logical ops into job: " << job_builder->job().job_conf().job_name()
            << ". Begin...\n";
  }

  HashSet<std::string> mut_op_names;
  const OpNode* first_node = subgraph_order.at(0);
  HashMap<std::string, OperatorConf> subgraph_op_name2conf;
  subgraph_op_name2conf.emplace(first_node->op().op_name(), first_node->op().op_conf());

  // add ctrl for strict order.
  for (int64_t i = 1; i < subgraph_order.size(); ++i) {
    const OpNode* this_node = subgraph_order.at(i);
    const OpNode* pre_node = subgraph_order.at(i - 1);
    const std::string& this_op_name = this_node->op().op_name();
    const std::string& pre_op_name = pre_node->op().op_name();
    CHECK(subgraph_op_name2conf.emplace(this_op_name, this_node->op().op_conf()).second);
    // build ctrl edge if need.
    if (!IsReachable(pre_op_name, this_op_name)) {
      subgraph_op_name2conf.at(this_op_name).add_ctrl_in_op_name(pre_op_name);
      mut_op_names.insert(this_op_name);
    }
  }

  std::vector<OperatorConf> nccl_op_confs;
  std::vector<ParallelConf> nccl_op_parallel_confs;
  // NOTE(chengcheng): ONLY support insert nccl to dst for memory.
  InsertNcclLogicalOpsAsCloseAsPossibleToDstNode(&subgraph_op_name2conf, &mut_op_names,
                                                 &nccl_op_confs, &nccl_op_parallel_confs,
                                                 subgraph_order, node2subgraph_order);

  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    VLOG(3) << " Try insert nccl logical ops into job: " << job_builder->job().job_conf().job_name()
            << ". ...End\n\n";
  }

  // NOTE(chengcheng): For NCCL logical correct exec order in pipeline multi-subgraph.
  do {
    if (nccl_op_confs.empty()) { break; }
    int64_t nccl_compute_stream_id = *stream_offset;
    if (nccl_compute_stream_id >= kMaxNcclComputeStreamCount) {
      break;  // NOTE(chengcheng): ONLY support kMaxNcclComputeStreamCount insert nccl subgraphs.
    }
    std::string stream_index_name = GetStreamIndexName(nccl_compute_stream_id);

    // NOTE(chengcheng): set ALL subgraph op and ALL nccl op stream index.
    for (auto& pair : subgraph_op_name2conf) {
      mut_op_names.insert(pair.first);
      pair.second.set_stream_name_hint(stream_index_name);
    }
    for (auto& nccl_op : nccl_op_confs) { nccl_op.set_stream_name_hint(stream_index_name); }
    (*stream_offset)++;
  } while (false);

  std::vector<OperatorConf> mut_op_confs;
  mut_op_confs.reserve(mut_op_names.size());
  for (const std::string& mut_op_name : mut_op_names) {
    mut_op_confs.emplace_back(subgraph_op_name2conf.at(mut_op_name));
  }
  job_builder->MutOpsOnlyOnce(mut_op_confs);

  CHECK_EQ(nccl_op_confs.size(), nccl_op_parallel_confs.size());
  for (int64_t i = 0; i < nccl_op_confs.size(); ++i) {
    CHECK_JUST(job_builder->AddOp(nccl_op_parallel_confs.at(i), nccl_op_confs.at(i)));
  }
}

void InsertBwSinkAccTickAndNcclLogicalOpsInPlacementGroupAfterAcc(
    const OpGraph& op_graph, JobBuilder* job_builder,
    const std::vector<const OpNode*>& ordered_acc_op_nodes,
    const HashMap<const OpNode*, int64_t>& op_node2global_order, const OpNode* bw_sink_op) {
  const OpNode* first_acc_op = ordered_acc_op_nodes.front();
  std::shared_ptr<const Shape> time_shape_before_acc = GetOpNodeFastestTimeShape(bw_sink_op);
  std::shared_ptr<const Shape> time_shape_after_acc = GetOpNodeFastestTimeShape(first_acc_op);
  VLOG(3) << " Find acc ops (num=" << ordered_acc_op_nodes.size()
          << ") in Job: " << job_builder->job().job_conf().job_name()
          << ", we will try insert special identity and ctrl for "
          << " UNSAFE handle ALL nccl ops between different time shape: "
          << time_shape_before_acc->DebugStr() << "->acc->" << time_shape_after_acc->DebugStr()
          << "\n\n";
  CHECK_GT(time_shape_before_acc->elem_cnt(), time_shape_after_acc->elem_cnt());
  CHECK_EQ(time_shape_before_acc->elem_cnt() % time_shape_after_acc->elem_cnt(), 0);

  for (const OpNode* acc : ordered_acc_op_nodes) {
    CHECK(SharedPtrShapeEqual(time_shape_before_acc, GetOpNodeInputTimeShape(acc)));
    CHECK(SharedPtrShapeEqual(time_shape_after_acc, GetOpNodeFastestTimeShape(acc)));
  }

  // NOTE(chengcheng): insert acc_tick after bw_sink_op, and this tick op conf will control
  //  after_acc_nccl_ops start.
  const auto& obns = bw_sink_op->op().output_bns();
  CHECK(!obns.empty());
  const std::string bw_sink_op_out_lbn =
      GenLogicalBlobName(bw_sink_op->op().BnInOp2Lbi(obns.Get(0)));
  VLOG(3) << " bw_sink_op : " << bw_sink_op->op().op_conf().DebugString();

  user_op::UserOpConfWrapper cast_to_tick_op =
      user_op::UserOpConfWrapperBuilder("System-CastToTick-" + NewUniqueId())
          .OpTypeName("cast_to_tick")
          .Input("in", bw_sink_op_out_lbn)
          .Output("out")
          .Build();

  OperatorConf bw_sink_acc_tick_conf;
  bw_sink_acc_tick_conf.set_name(std::string("System-BwSinkTick-AccTick_") + NewUniqueId());
  auto* acc_conf = bw_sink_acc_tick_conf.mutable_acc_tick_conf();
  acc_conf->set_one(cast_to_tick_op.output("out", 0));
  acc_conf->set_acc("acc");
  acc_conf->set_max_acc_num(time_shape_before_acc->elem_cnt() / time_shape_after_acc->elem_cnt());

  OperatorConf bw_sink_final_tick_conf;
  bw_sink_final_tick_conf.set_name(std::string("System-BwSinkFinalTick-DeviceTick_")
                                   + NewUniqueId());
  auto* tick_conf = bw_sink_final_tick_conf.mutable_device_tick_conf();
  tick_conf->add_tick(GenLogicalBlobName(bw_sink_acc_tick_conf.name(), "acc"));
  tick_conf->set_out("out");

  // insert nccl ops after acc
  std::vector<OperatorConf> after_acc_nccl_op_confs;
  std::vector<ParallelConf> after_acc_nccl_parallel_confs;
  HashMap<std::string, OperatorConf> mut_consumer_name2op;

  InsertNcclLogicalOpsAfterAcc(op_graph, op_node2global_order, ordered_acc_op_nodes,
                               bw_sink_final_tick_conf.name(), &mut_consumer_name2op,
                               &after_acc_nccl_op_confs, &after_acc_nccl_parallel_confs);

  if (after_acc_nccl_op_confs.empty()) {
    CHECK(after_acc_nccl_parallel_confs.empty());
    CHECK(mut_consumer_name2op.empty());
  } else {
    // insert bw sink acc tick ops
    CHECK_JUST(
        job_builder->AddOp(bw_sink_op->parallel_desc().parallel_conf(), cast_to_tick_op.op_conf()));
    VLOG(3) << " Insert cast_to_tick_op : " << cast_to_tick_op.op_conf().DebugString();

    CHECK_JUST(
        job_builder->AddOp(bw_sink_op->parallel_desc().parallel_conf(), bw_sink_acc_tick_conf));
    VLOG(3) << " Insert bw_sink_acc_tick_op : " << bw_sink_acc_tick_conf.DebugString();

    CHECK_JUST(
        job_builder->AddOp(bw_sink_op->parallel_desc().parallel_conf(), bw_sink_final_tick_conf));
    VLOG(3) << " Insert bw_sink_final_tick_op : " << bw_sink_final_tick_conf.DebugString();

    // insert nccl ops after acc
    for (const auto& pair : mut_consumer_name2op) {
      CHECK_JUST(job_builder->MutOpOnlyOnce(pair.second));
    }
    CHECK_EQ(after_acc_nccl_op_confs.size(), after_acc_nccl_parallel_confs.size());
    for (int64_t i = 0; i < after_acc_nccl_op_confs.size(); ++i) {
      CHECK_JUST(
          job_builder->AddOp(after_acc_nccl_parallel_confs.at(i), after_acc_nccl_op_confs.at(i)));
    }
  }
}



}  // namespace

REGISTER_JOB_PASS("LogicalChainPass", LogicalChainPass);

}  // namespace oneflow
