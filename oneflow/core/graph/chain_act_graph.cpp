#include "oneflow/core/graph/chain_act_graph.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

inline double Duration4RegstActConsumerPath(const RegstActCtx& regst_act_ctx,
                                            const ActEvent* consumer_act_event,
                                            const ChainActNode* consumer_node) {
  return regst_act_ctx.node2duration_to_producer.at(consumer_node) + consumer_act_event->stop_time()
         - regst_act_ctx.regst_act->producer_act_event->start_time();
}

ChainActNode::ChainActNode(std::list<std::unique_ptr<ActEvent>>&& act_events)
    : act_events_(std::move(act_events)) {
  act_events_.sort([](const std::unique_ptr<ActEvent>& lhs, const std::unique_ptr<ActEvent>& rhs) {
    return lhs->start_time() < rhs->start_time();
  });
  double pre_stop_time = 0;
  for (auto& act_event : act_events_) {
    double duration = Duration4ActEvent(*act_event);
    act_event->set_start_time(pre_stop_time);
    act_event->set_stop_time(pre_stop_time + duration);
    pre_stop_time = act_event->stop_time();
  }
}

void ChainActNode::ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const {
  for (const auto& act_event : act_events_) { Handler(act_event.get()); }
}

void ChainActNode::ForEachProducedRegstAct(
    const std::function<void(const RegstAct*)>& Handler) const {
  for (const auto& produced_regst_act : produced_regst_acts_) { Handler(produced_regst_act.get()); }
}

void ChainActNode::ForEachLastConsumedRegstAct(
    const std::function<void(const RegstAct*)>& Handler) const {
  for (const auto& last_consumed_regst_act : last_consumed_regst_acts_) {
    Handler(last_consumed_regst_act);
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
  ForEachNode([&](const ChainActNode* node) {
    node->ForEachProducedRegstAct([&](const RegstAct* regst_act) {
      int64_t produced_cnt = ++regst_desc_id2produced_cnt[regst_act->regst_desc_id];
      if (max_cnt < produced_cnt) { max_cnt = produced_cnt; }
      for (const ActEvent* act_event : regst_act->consumer_act_events) {
        std::pair<int64_t, int64_t> consumed_regst_desc_id(regst_act->regst_desc_id,
                                                           act_event->actor_id());
        int64_t used_cnt = ++regst_desc_id_consumed2used_cnt[consumed_regst_desc_id];
        if (max_cnt < used_cnt) { max_cnt = used_cnt; }
      }
    });
  });
  for (const auto& pair : regst_desc_id_consumed2used_cnt) {
    uint64_t produced_cnt = regst_desc_id2produced_cnt.at(pair.first.first);
    Handler(pair.first.first, pair.first.second,
            1.0 * max_cnt / std::min(produced_cnt, pair.second));
  }
}

void ChainActGraph::ForEachRegstActConsumerPathDuration(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  HashSet<std::shared_ptr<RegstActCtx>> regst_act_ctx_window;
  HashMap<const RegstAct*, std::shared_ptr<RegstActCtx>> regst_act2regst_act_ctx;
  TopoForEachChainActNode([&](const ChainActNode* cur_node) {
    for (auto& regst_act_ctx : regst_act_ctx_window) {
      const auto& fake_producer_outs = regst_act_ctx->regst_act->fake_producer_outs;
      if (fake_producer_outs.find(cur_node) != fake_producer_outs.end()) { continue; }
      CalcRegstActNodePathDuration(regst_act_ctx, cur_node);
    }
    cur_node->ForEachProducedRegstAct([&](const RegstAct* regst_act) {
      const ChainActNode* producer = Node4ActEvent(regst_act->producer_act_event);
      std::shared_ptr<RegstActCtx> regst_act_ctx(new RegstActCtx(regst_act, producer));
      regst_act_ctx_window.insert(regst_act_ctx);
      regst_act2regst_act_ctx[regst_act] = regst_act_ctx;
    });
    cur_node->ForEachLastConsumedRegstAct([&](const RegstAct* regst_act) {
      for (const ActEvent* consumer_act_event : regst_act->consumer_act_events) {
        double duration =
            Duration4RegstActConsumerPath(*(regst_act2regst_act_ctx.at(regst_act)),
                                          consumer_act_event, Node4ActEvent(consumer_act_event));
        Handler(regst_act->regst_desc_id, consumer_act_event->actor_id(), duration);
      }
      regst_act_ctx_window.erase(regst_act2regst_act_ctx.at(regst_act));
      regst_act2regst_act_ctx.erase(regst_act);
    });
  });
}

void ChainActGraph::CalcRegstActNodePathDuration(std::shared_ptr<RegstActCtx> regst_act_ctx,
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

void ChainActGraph::InitTaskId2TaskProto() {
  for (const auto& task_proto : plan_->task()) {
    task_id2task_proto_.emplace(task_proto.task_id(), &task_proto);
  }
}

void ChainActGraph::InitNodes(
    std::list<std::unique_ptr<ActEvent>>&& act_events,
    HashMap<std::pair<int64_t, int64_t>, const ActEvent*>* regst_uid2producer_act_event) {
  HashMap<std::string, std::list<std::unique_ptr<ActEvent>>> chain_with_act_id_str2act_events;
  for (auto& act_event : act_events) {
    int64_t actor_id = act_event->actor_id();
    int64_t act_id = act_event->act_id();
    const TaskProto& task_proto = GetTaskProto(actor_id);
    int64_t chain_id = task_proto.task_set_info().chain_id();
    chain_with_act_id_str2act_events[std::to_string(chain_id) + ":" + std::to_string(act_id)]
        .push_back(std::move(act_event));
  }
  for (auto& pair : chain_with_act_id_str2act_events) {
    ChainActNode* chain_act_node = new ChainActNode(std::move(pair.second));
    AddAllocatedNode(chain_act_node);
    chain_act_node->ForEachActEvent([&](const ActEvent* act_event) {
      int64_t actor_id = act_event->actor_id();
      int64_t act_id = act_event->act_id();
      const TaskProto& task_proto = GetTaskProto(actor_id);
      act_event2chain_node_.emplace(act_event, chain_act_node);
      for (const auto& produced_regst_desc : task_proto.produced_regst_desc()) {
        int64_t regst_desc_id = produced_regst_desc.second.regst_desc_id();
        std::pair<int64_t, int64_t> regst_uid(regst_desc_id, act_id);
        regst_uid2producer_act_event->emplace(regst_uid, act_event);
      }
    });
  }
}

void ChainActGraph::InitEdges(
    const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
    HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>*
        regst_uid2consumer_act_events) {
  ForEachNode([&](const ChainActNode* node) {
    HashMap<ChainActNode*, double> producer2max_stop_time;
    ChainActNode* consumer_node = nullptr;
    node->ForEachActEvent([&](const ActEvent* consumer_act_event) {
      consumer_node = act_event2chain_node_.at(consumer_act_event);
      for (const auto& readable : consumer_act_event->readable_regst_infos()) {
        std::pair<int64_t, int64_t> regst_uid(readable.regst_desc_id(), readable.act_id());
        const auto& producer_act_event_it = regst_uid2producer_act_event.find(regst_uid);
        if (producer_act_event_it == regst_uid2producer_act_event.end()) { continue; }
        (*regst_uid2consumer_act_events)[regst_uid].push_back(consumer_act_event);
        act_event_with_consumer_.emplace(producer_act_event_it->second);
        ChainActNode* producer = act_event2chain_node_.at(producer_act_event_it->second);
        if (producer == node) { continue; }
        double& max_stop_time = producer2max_stop_time[producer];
        max_stop_time = std::max(max_stop_time, producer_act_event_it->second->stop_time());
      }
    });
    for (const auto& pair : producer2max_stop_time) {
      ChainActEdge* edge = new ChainActEdge(pair.second);
      AddAllocatedEdge(edge);
      Connect(pair.first, edge, consumer_node);
    }
  });
}

void ChainActGraph::InitNodeProducedRegstActs(
    const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
    const HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>&
        regst_uid2consumer_act_events) {
  for (const auto& pair : regst_uid2producer_act_event) {
    const auto& consumers_act_event_it = regst_uid2consumer_act_events.find(pair.first);
    if (consumers_act_event_it == regst_uid2consumer_act_events.end()) { continue; }
    ChainActNode* producer = act_event2chain_node_.at(pair.second);
    auto regst_act =
        std::make_unique<RegstAct>(pair.first.first, pair.second, consumers_act_event_it->second);
    ForEachOutEdge(producer, [&](const ChainActEdge* out_edge) {
      if (out_edge->duration() < regst_act->producer_act_event->stop_time()) {
        regst_act->fake_producer_outs.emplace(out_edge->dst_node());
      }
    });
    producer->AddProducedRegstActs(std::move(regst_act));
  }
}

void ChainActGraph::InitNodeLastConsumedRegstActs() {
  auto TopoOrderValue4Node = MakeGetterTopoOrderValue4Node();
  ForEachNode([&](const ChainActNode* node) {
    node->ForEachProducedRegstAct([&](const RegstAct* regst_act) {
      int64_t max_consumer_topo_order_value = -1;
      const ActEvent* last_consumer_act_event = nullptr;
      for (const ActEvent* consumer_act_event : regst_act->consumer_act_events) {
        int64_t cur_consumer_topo_order_value =
            TopoOrderValue4Node(Node4ActEvent(consumer_act_event));
        if (cur_consumer_topo_order_value > max_consumer_topo_order_value) {
          max_consumer_topo_order_value = cur_consumer_topo_order_value;
          last_consumer_act_event = consumer_act_event;
        }
      }
      act_event2chain_node_.at(last_consumer_act_event)->AddLastConsumedRegstActs(regst_act);
    });
  });
}

std::function<int64_t(const ChainActNode*)> ChainActGraph::MakeGetterTopoOrderValue4Node() const {
  auto node2topo_order_value = std::make_shared<HashMap<const ChainActNode*, int64_t>>();
  int64_t topo_order_value = -1;
  TopoForEachChainActNode([&](const ChainActNode* chain_act_node) {
    (*node2topo_order_value)[chain_act_node] = ++topo_order_value;
  });
  return
      [node2topo_order_value](const ChainActNode* node) { return node2topo_order_value->at(node); };
}

const ChainActNode* ChainActGraph::Node4ActEvent(const ActEvent* act_event) const {
  return act_event2chain_node_.at(act_event);
}

void ChainActGraph::TopoForEachChainActNode(
    const std::function<void(const ChainActNode*)>& Handler) const {
  std::list<const ChainActNode*> starts;
  ForEachNode([&](const ChainActNode* node) {
    if (node->in_edges().empty()) { starts.push_back(node); }
  });
  starts.sort([](const ChainActNode* lhs, const ChainActNode* rhs) {
    return lhs->act_id() > rhs->act_id();
  });
  DfsTopoForEachNode(starts, &ChainActNode::ForEachNodeOnInEdge,
                     &ChainActNode::ForEachNodeOnOutEdge, Handler);
}

void ChainActGraph::ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const {
  ForEachNode([&](const ChainActNode* node) { node->ForEachActEvent(Handler); });
}

void ChainActGraph::ForEachInEdge(const ChainActNode* node,
                                  const std::function<void(const ChainActEdge*)>& Handler) const {
  for (const ChainActEdge* in_edge : node->in_edges()) { Handler(in_edge); }
}

void ChainActGraph::ForEachOutEdge(const ChainActNode* node,
                                   const std::function<void(const ChainActEdge*)>& Handler) const {
  for (const ChainActEdge* out_edge : node->out_edges()) { Handler(out_edge); }
}

ChainActGraph::ChainActGraph(const Plan& plan, std::list<std::unique_ptr<ActEvent>>&& act_events)
    : plan_(&plan) {
  HashMap<std::pair<int64_t, int64_t>, const ActEvent*> regst_uid2producer_act_event;
  HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>> regst_uid2consumer_act_events;
  InitTaskId2TaskProto();
  InitNodes(std::move(act_events), &regst_uid2producer_act_event);
  InitEdges(regst_uid2producer_act_event, &regst_uid2consumer_act_events);
  InitNodeProducedRegstActs(regst_uid2producer_act_event, regst_uid2consumer_act_events);
  InitNodeLastConsumedRegstActs();
}

}  // namespace oneflow
