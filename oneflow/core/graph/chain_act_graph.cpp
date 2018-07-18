#include "oneflow/core/graph/chain_act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {
ChainActNode::ChainActNode(std::list<ActEvent*> act_events) {
  act_events.sort(
      [](ActEvent* lhs, ActEvent* rhs) { return lhs->start_time() < rhs->start_time(); });
  double pre_stop_time = 0;
  for (ActEvent* act_event : act_events) {
    double duration = Duration4ActEvent(*act_event);
    act_event->set_start_time(pre_stop_time);
    act_event->set_stop_time(pre_stop_time + duration);
    pre_stop_time = act_event->stop_time();
    const ActEvent* const_act_event = act_event;
    act_events_.push_back(const_act_event);
  }
}

void ChainActGraph::ForEachRegstDescConsumerPathMeanDuration(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  std::map<std::pair<int64_t, int64_t>, double> regst_desc_id_consumed2duration;
  std::map<std::pair<int64_t, int64_t>, int> regst_desc_id_consumed2cnt;
  ForEachRegstActConsumerPathDuration(
      [&](int64_t regst_desc_id, int64_t consumer_actor_id, double duration) {
        std::pair<int64_t, int64_t> regst_desc_id_consumed(regst_desc_id, consumer_actor_id);
        regst_desc_id_consumed2duration[regst_desc_id_consumed] += duration;
        ++regst_desc_id_consumed2cnt[regst_desc_id_consumed];
      });
  for (const auto& pair : regst_desc_id_consumed2duration) {
    Handler(pair.first.first, pair.first.second,
            pair.second / regst_desc_id_consumed2cnt.at(pair.first));
  }
}

void ChainActGraph::ForEachRegstDescConsumerPathIIScale(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  std::map<std::pair<int64_t, int64_t>, uint64_t> regst_desc_id_consumed2used_cnt;
  std::map<int64_t, uint64_t> regst_desc_id2produced_cnt;
  uint64_t max_cnt = 0;
  for (const auto& pair : regst_uid2regst_act_) {
    if (pair.second->consumer_act_events.empty()) { continue; }
    int64_t regst_desc_id = pair.second->regst_desc_id;
    int64_t produced_cnt = ++regst_desc_id2produced_cnt[regst_desc_id];
    if (max_cnt < produced_cnt) { max_cnt = produced_cnt; }
    for (const ActEvent* act_event : pair.second->consumer_act_events) {
      std::pair<int64_t, int64_t> consumed_regst_desc_id(regst_desc_id, act_event->actor_id());
      int64_t used_cnt = ++regst_desc_id_consumed2used_cnt[consumed_regst_desc_id];
      if (max_cnt < used_cnt) { max_cnt = used_cnt; }
    }
  }
  for (const auto& pair : regst_desc_id_consumed2used_cnt) {
    uint64_t produced_cnt = regst_desc_id2produced_cnt.at(pair.first.first);
    Handler(pair.first.first, pair.first.second,
            1.0 * max_cnt / std::min(produced_cnt, pair.second));
  }
}

void ChainActGraph::ForEachRegstActConsumerPathDuration(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  HashSet<RegstActCtx*> regst_act_ctx_window;
  HashMap<const RegstAct*, RegstActCtx*> regst_act2regst_act_ctx;
  for (int64_t i = 0; i < topo_nodes_.size(); ++i) {
    const ChainActNode* cur_node = topo_nodes_.at(i);
    for (RegstActCtx* regst_act_ctx : regst_act_ctx_window) {
      const auto& actual_producer_outs = regst_act_ctx->regst_act->actual_producer_outs;
      if (actual_producer_outs.find(cur_node) == actual_producer_outs.end()) { continue; }
      ForEachRegstActNodePathDuration(regst_act_ctx, cur_node);
    }
    for (const RegstAct* regst_act : cur_node->produced_regst_acts()) {
      const ChainActNode* producer = Node4ActEvent(regst_act->producer_act_event);
      RegstActCtx* regst_act_ctx = new RegstActCtx;
      regst_act_ctx->regst_act = regst_act;
      regst_act_ctx->node2duration_to_producer[producer] = 0;
      regst_act_ctx_window.insert(regst_act_ctx);
      regst_act2regst_act_ctx[regst_act] = regst_act_ctx;
    }
    for (const RegstAct* regst_act : cur_node->last_consumed_regst_acts()) {
      ForEachConsumerPathDuration(regst_act2regst_act_ctx.at(regst_act), Handler);
      regst_act_ctx_window.erase(regst_act2regst_act_ctx.at(regst_act));
      delete regst_act2regst_act_ctx.at(regst_act);
    }
  }
}

void ChainActGraph::ForEachRegstActNodePathDuration(RegstActCtx* regst_act_ctx,
                                                    const ChainActNode* node) const {
  double duration = 0;
  ForEachInEdge(node, [&](const ChainActEdge* in_edge) {
    const ChainActNode* in_node = in_edge->src_node();
    if (regst_act_ctx->node2duration_to_producer.find(in_node)
        != regst_act_ctx->node2duration_to_producer.end()) {
      duration = std::max(
          duration, in_edge->duration() + regst_act_ctx->node2duration_to_producer.at(in_node));
    }
  });
  if (duration > 0) { regst_act_ctx->node2duration_to_producer[node] = duration; }
}

void ChainActGraph::ForEachConsumerPathDuration(
    RegstActCtx* regst_act_ctx,
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  const RegstAct* regst_act = regst_act_ctx->regst_act;
  for (const ActEvent* consumer_act_event : regst_act->consumer_act_events) {
    double duration = regst_act_ctx->node2duration_to_producer.at(Node4ActEvent(consumer_act_event))
                      + consumer_act_event->stop_time()
                      - regst_act->producer_act_event->start_time();
    Handler(regst_act->regst_desc_id, consumer_act_event->actor_id(), duration);
  }
}

void ChainActGraph::InitTaskId2TaskProto() {
  for (const auto& task_proto : plan_->task()) {
    task_id2task_proto_.emplace(task_proto.task_id(), &task_proto);
  }
}

void ChainActGraph::TopoForEachChainActNode(
    std::list<ChainActNode*>& starts, const std::function<void(ChainActNode*)>& Handler) const {
  starts.sort([](const ChainActNode* lhs, const ChainActNode* rhs) {
    return lhs->act_id() > rhs->act_id();
  });
  DfsTopoForEachNode(starts, &ChainActNode::ForEachNodeOnInEdge,
                     &ChainActNode::ForEachNodeOnOutEdge, Handler);
}

void ChainActGraph::InitNodes() {
  HashMap<std::string, std::list<ActEvent*>> chain_id7act_id_str2act_events;
  for (ActEvent& act_event : *act_events_) {
    int64_t actor_id = act_event.actor_id();
    int64_t act_id = act_event.act_id();
    const TaskProto& task_proto = GetTaskProto(actor_id);
    int64_t chain_id = task_proto.task_set_info().chain_id();
    chain_id7act_id_str2act_events[std::to_string(chain_id) + ":" + std::to_string(act_id)]
        .push_back(&act_event);
  }
  for (auto node_str : chain_id7act_id_str2act_events) {
    ChainActNode* chain_act_node = new ChainActNode(node_str.second);
    AddAllocatedNode(chain_act_node);
    for (const ActEvent* act_event : chain_act_node->act_events()) {
      int64_t actor_id = act_event->actor_id();
      int64_t act_id = act_event->act_id();
      const TaskProto& task_proto = GetTaskProto(actor_id);
      act_event2chain_node_.emplace(act_event, chain_act_node);
      for (const auto& pair : task_proto.produced_regst_desc()) {
        int64_t regst_desc_id = pair.second.regst_desc_id();
        std::pair<int64_t, int64_t> regst_uid(regst_desc_id, act_id);
        RegstAct* regst_act = new RegstAct;
        regst_act->regst_desc_id = regst_desc_id;
        regst_act->producer_act_event = act_event;
        regst_uid2regst_act_.emplace(regst_uid, regst_act);
      }
    }
  }
}

void ChainActGraph::InitEdges() {
  ForEachNode([&](ChainActNode* node) {
    HashMap<ChainActNode*, double> producer_node2max_stop_time;
    for (const ActEvent* act_event : node->act_events()) {
      for (const auto& readable : act_event->readable_regst_infos()) {
        std::pair<int64_t, int64_t> regst_uid(readable.regst_desc_id(), readable.act_id());
        const auto& regst_act_it = regst_uid2regst_act_.find(regst_uid);
        if (regst_act_it == regst_uid2regst_act_.end()) { continue; }
        act_event_has_consumer_.insert(regst_act_it->second->producer_act_event);
        regst_act_it->second->consumer_act_events.push_back(act_event);
        ChainActNode* producer_node =
            act_event2chain_node_.at(regst_act_it->second->producer_act_event);
        if (producer_node == node) { continue; }
        double& max_stop_time = producer_node2max_stop_time[producer_node];
        max_stop_time =
            std::max(max_stop_time, regst_act_it->second->producer_act_event->stop_time());
      }
    }
    for (const auto& pair : producer_node2max_stop_time) {
      ChainActEdge* edge = NewEdge();
      Connect(pair.first, edge, node);
      edge->set_duration(pair.second);
    }
  });
}

void ChainActGraph::InitTopoOrderValue() {
  std::list<ChainActNode*> sources;
  ForEachNode([&](ChainActNode* node) {
    if (node->in_edges().empty()) { sources.push_back(node); }
  });
  int64_t topo_order_value = -1;
  TopoForEachChainActNode(sources, [&](ChainActNode* chain_act_node) {
    chain_act_node->set_topo_order_value(++topo_order_value);
    topo_nodes_.push_back(chain_act_node);
  });
}

void ChainActGraph::InitRegstActProduced7LastConsumedNode() {
  for (auto& pair : regst_uid2regst_act_) {
    RegstAct* regst_act = pair.second;
    if (regst_act->consumer_act_events.empty()) { continue; }
    ChainActNode* producer = act_event2chain_node_.at(regst_act->producer_act_event);
    ForEachOutEdge(producer, [&](const ChainActEdge* out_edge) {
      if (out_edge->duration() >= regst_act->producer_act_event->stop_time()) {
        regst_act->actual_producer_outs.insert(out_edge->dst_node());
      }
    });
    producer->set_produced_regst_acts(regst_act);
    int64_t max_consumer_topo_order_value = -1;
    const ActEvent* last_consumer_act_event = nullptr;
    for (const ActEvent* consumer_act_event : regst_act->consumer_act_events) {
      if (Node4ActEvent(consumer_act_event)->topo_order_value() > max_consumer_topo_order_value) {
        max_consumer_topo_order_value = Node4ActEvent(consumer_act_event)->topo_order_value();
        last_consumer_act_event = consumer_act_event;
      }
    }
    act_event2chain_node_.at(last_consumer_act_event)->set_last_consumed_regst_acts(regst_act);
  }
}

void ChainActGraph::ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const {
  for (const ActEvent& act_event : *act_events_) { Handler(&act_event); }
}

void ChainActGraph::ForEachInEdge(const ChainActNode* node,
                                  const std::function<void(const ChainActEdge*)>& Handler) const {
  for (const ChainActEdge* in_edge : node->in_edges()) { Handler(in_edge); }
}

void ChainActGraph::ForEachOutEdge(const ChainActNode* node,
                                   const std::function<void(const ChainActEdge*)>& Handler) const {
  for (const ChainActEdge* out_edge : node->out_edges()) { Handler(out_edge); }
}

ChainActGraph::ChainActGraph(const Plan& plan, std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  InitTaskId2TaskProto();
  InitNodes();
  InitEdges();
  InitTopoOrderValue();
  InitRegstActProduced7LastConsumedNode();
}

}  // namespace oneflow
