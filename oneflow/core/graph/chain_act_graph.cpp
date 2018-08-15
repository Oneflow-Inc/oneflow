#include "oneflow/core/graph/chain_act_graph.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

inline double Duration4RegstActConsumerPath(const ActEvent* producer_act_event,
                                            const ActEvent* consumer_act_event,
                                            double path_duration) {
  return path_duration + consumer_act_event->stop_time() - producer_act_event->start_time();
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

void ChainActNode::ForEachInEdge(const std::function<void(const ChainActEdge*)>& Handler) const {
  for (const ChainActEdge* in_edge : in_edges()) { Handler(in_edge); }
}

void ChainActNode::ForEachOutEdge(const std::function<void(const ChainActEdge*)>& Handler) const {
  for (const ChainActEdge* out_edge : out_edges()) { Handler(out_edge); }
}

void ChainActNode::ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const {
  for (const auto& act_event : act_events_) { Handler(act_event.get()); }
}

void ChainActNode::ForEachProducedRegstActGroup(
    const std::function<void(const std::list<const RegstAct*>&)>& Handler) const {
  for (const auto& pair : fake_outs2produced_regst_act_group_) { Handler(pair.second); }
}

void ChainActNode::ForEachLastConsumedRegstAct(
    const std::function<void(const RegstAct*)>& Handler) const {
  for (const auto& last_consumed_regst_act_group : last_consumed_regst_act_groups_) {
    for (const RegstAct* regst_act : last_consumed_regst_act_group) { Handler(regst_act); }
  }
}

void ChainActNode::AddProducedRegstAct(std::unique_ptr<RegstAct>&& regst_act) {
  fake_outs2produced_regst_act_group_[regst_act->fake_producer_outs].push_back(regst_act.get());
  produced_regst_acts_.push_back(std::move(regst_act));
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
    node->ForEachLastConsumedRegstAct([&](const RegstAct* regst_act) {
      int64_t produced_cnt = ++regst_desc_id2produced_cnt[regst_act->regst_desc_id];
      if (max_cnt < produced_cnt) { max_cnt = produced_cnt; }
      for (const ActEvent* act_event : regst_act->consumer_act_events) {
        std::pair<int64_t, int64_t> regst_desc_id_consumed(regst_act->regst_desc_id,
                                                           act_event->actor_id());
        int64_t used_cnt = ++regst_desc_id_consumed2used_cnt[regst_desc_id_consumed];
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

double ChainActGraph::CalcBaseII() const {
  int64_t max_act_cnt = 0;
  HashMap<int64_t, int64_t> actor_id2outputed_act_cnt;
  ForEachActEvent([&](const ActEvent* act_event) {
    int64_t actor_id = act_event->actor_id();
    if (IsActEventWithConsumer(act_event)) {
      ++actor_id2outputed_act_cnt[actor_id];
      max_act_cnt = std::max(max_act_cnt, actor_id2outputed_act_cnt[actor_id]);
    }
  });
  HashMap<int64_t, double> stream_id2total_calc_time;
  ForEachActEvent([&](const ActEvent* act_event) {
    int64_t actor_id = act_event->actor_id();
    auto frequence_it = actor_id2outputed_act_cnt.find(actor_id);
    if (frequence_it == actor_id2outputed_act_cnt.end()) { return; }
    int64_t stream_id = act_event->work_stream_id();
    stream_id2total_calc_time[stream_id] += Duration4ActEvent(*act_event);
  });
  double base_ii = 0;
  for (const auto& pair : stream_id2total_calc_time) {
    base_ii = std::max(base_ii, pair.second / max_act_cnt);
  }
  return base_ii;
}

void ChainActGraph::ForEachRegstActConsumerPathDuration(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  HashSet<std::shared_ptr<RegstActGroupCtx>> ctx_window;
  HashMap<const RegstAct*, std::shared_ptr<RegstActGroupCtx>> regst_act2ctx;
  TopoForEachChainActNode([&](const ChainActNode* cur_node) {
    for (auto& ctx : ctx_window) {
      const auto& fake_producer_outs = ctx->regst_act_group.front()->fake_producer_outs;
      if (fake_producer_outs.find(cur_node) != fake_producer_outs.end()) { continue; }
      CalcRegstActNodePathDuration(ctx.get(), cur_node);
    }
    cur_node->ForEachProducedRegstActGroup([&](const std::list<const RegstAct*>& regst_act_group) {
      std::shared_ptr<RegstActGroupCtx> ctx(new RegstActGroupCtx(regst_act_group, cur_node));
      CHECK(ctx_window.emplace(ctx).second);
      for (const RegstAct* regst_act : regst_act_group) {
        CHECK(regst_act2ctx.emplace(regst_act, ctx).second);
      }
    });
    cur_node->ForEachLastConsumedRegstAct([&](const RegstAct* regst_act) {
      const ActEvent* producer = regst_act->producer_act_event;
      for (const ActEvent* consumer : regst_act->consumer_act_events) {
        double path_duration =
            regst_act2ctx.at(regst_act)->node2duration_to_producer.at(Node4ActEvent(consumer));
        Handler(regst_act->regst_desc_id, consumer->actor_id(),
                Duration4RegstActConsumerPath(producer, consumer, path_duration));
      }
      ctx_window.erase(regst_act2ctx.at(regst_act));
      regst_act2ctx.erase(regst_act);
    });
  });
}

void ChainActGraph::CalcRegstActNodePathDuration(RegstActGroupCtx* regst_act_group_ctx,
                                                 const ChainActNode* node) const {
  double duration = 0;
  node->ForEachInEdge([&](const ChainActEdge* in_edge) {
    const ChainActNode* in_node = in_edge->src_node();
    if (regst_act_group_ctx->node2duration_to_producer.find(in_node)
        != regst_act_group_ctx->node2duration_to_producer.end()) {
      duration =
          std::max(duration, in_edge->duration()
                                 + regst_act_group_ctx->node2duration_to_producer.at(in_node));
    }
  });
  if (duration > 0) { regst_act_group_ctx->node2duration_to_producer[node] = duration; }
}

void ChainActGraph::InitTaskId2TaskProto() {
  for (const auto& task_proto : plan_->task()) {
    CHECK(task_id2task_proto_.emplace(task_proto.task_id(), &task_proto).second);
  }
}

void ChainActGraph::InitNodes(
    std::list<std::unique_ptr<ActEvent>>&& act_events,
    HashMap<std::pair<int64_t, int64_t>, const ActEvent*>* regst_uid2producer_act_event) {
  HashMap<std::pair<int64_t, int64_t>, std::list<std::unique_ptr<ActEvent>>>
      chain_id_with_act_id2act_events;
  for (auto& act_event : act_events) {
    int64_t actor_id = act_event->actor_id();
    int64_t act_id = act_event->act_id();
    const TaskProto& task_proto = GetTaskProto(actor_id);
    int64_t chain_id = task_proto.task_set_info().chain_id();
    std::pair<int64_t, int64_t> chain_act_id_pair(chain_id, act_id);
    chain_id_with_act_id2act_events[chain_act_id_pair].push_back(std::move(act_event));
  }
  for (auto& pair : chain_id_with_act_id2act_events) {
    ChainActNode* chain_act_node = new ChainActNode(std::move(pair.second));
    AddAllocatedNode(chain_act_node);
    chain_act_node->ForEachActEvent([&](const ActEvent* act_event) {
      int64_t actor_id = act_event->actor_id();
      int64_t act_id = act_event->act_id();
      const TaskProto& task_proto = GetTaskProto(actor_id);
      CHECK(act_event2chain_node_.emplace(act_event, chain_act_node).second);
      for (const auto& produced_regst_desc : task_proto.produced_regst_desc()) {
        int64_t regst_desc_id = produced_regst_desc.second.regst_desc_id();
        std::pair<int64_t, int64_t> regst_uid(regst_desc_id, act_id);
        CHECK(regst_uid2producer_act_event->emplace(regst_uid, act_event).second);
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
    ChainActNode* mut_node = nullptr;
    node->ForEachActEvent([&](const ActEvent* consumer_act_event) {
      mut_node = act_event2chain_node_.at(consumer_act_event);
      CHECK_EQ(node, mut_node);
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
      Connect(pair.first, edge, mut_node);
    }
  });
}

void ChainActGraph::InitNodeProducedRegstAct(
    const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
    const HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>&
        regst_uid2consumer_act_events) const {
  for (const auto& pair : regst_uid2producer_act_event) {
    const auto& consumers_act_event_it = regst_uid2consumer_act_events.find(pair.first);
    if (consumers_act_event_it == regst_uid2consumer_act_events.end()) { continue; }
    ChainActNode* producer = act_event2chain_node_.at(pair.second);
    auto regst_act =
        std::make_unique<RegstAct>(pair.first.first, pair.second, consumers_act_event_it->second);
    producer->ForEachOutEdge([&](const ChainActEdge* out_edge) {
      if (out_edge->duration() < regst_act->producer_act_event->stop_time()) {
        CHECK(regst_act->fake_producer_outs.emplace(out_edge->dst_node()).second);
      }
    });
    producer->AddProducedRegstAct(std::move(regst_act));
  }
}

void ChainActGraph::InitNodeLastConsumedRegstActGroup() const {
  auto TopoOrderValue4Node = MakeGetterTopoOrderValue4Node();
  auto ForEachConsumer = [](const std::list<const RegstAct*>& regst_act_group,
                            const std::function<void(const ActEvent*)>& Handler) {
    for (const RegstAct* regst_act : regst_act_group) {
      for (const ActEvent* cur_consumer : regst_act->consumer_act_events) { Handler(cur_consumer); }
    }
  };
  ForEachNode([&](const ChainActNode* node) {
    node->ForEachProducedRegstActGroup([&](const std::list<const RegstAct*>& regst_act_group) {
      int64_t max_topo_order_value = -1;
      const ActEvent* last_consumer = nullptr;
      ForEachConsumer(regst_act_group, [&](const ActEvent* cur_consumer) {
        int64_t cur_topo_order_value = TopoOrderValue4Node(Node4ActEvent(cur_consumer));
        if (cur_topo_order_value > max_topo_order_value) {
          max_topo_order_value = cur_topo_order_value;
          last_consumer = cur_consumer;
        }
      });
      act_event2chain_node_.at(last_consumer)->AddLastConsumedRegstActGroup(regst_act_group);
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

bool ChainActGraph::IsActEventWithConsumer(const ActEvent* act_event) const {
  return act_event_with_consumer_.find(act_event) != act_event_with_consumer_.end();
}

ChainActGraph::ChainActGraph(const Plan& plan, std::list<std::unique_ptr<ActEvent>>&& act_events)
    : plan_(&plan) {
  HashMap<std::pair<int64_t, int64_t>, const ActEvent*> regst_uid2producer_act_event;
  HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>> regst_uid2consumer_act_events;
  InitTaskId2TaskProto();
  InitNodes(std::move(act_events), &regst_uid2producer_act_event);
  InitEdges(regst_uid2producer_act_event, &regst_uid2consumer_act_events);
  InitNodeProducedRegstAct(regst_uid2producer_act_event, regst_uid2consumer_act_events);
  InitNodeLastConsumedRegstActGroup();
}

}  // namespace oneflow
