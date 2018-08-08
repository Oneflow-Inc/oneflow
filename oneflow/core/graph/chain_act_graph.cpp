#include "oneflow/core/graph/chain_act_graph.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

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

void ChainActNode::ForEachStartConsumerRegstInfos(
    const std::function<void(const ConsumerRegstInfos*)>& Handler) const {
  for (const auto& start_consumed_regst_infos : start_consumed_regst_infos_) {
    Handler(start_consumed_regst_infos.get());
  }
}

void ChainActNode::ForEachStopConsumerRegstInfos(
    const std::function<void(const ConsumerRegstInfos*)>& Handler) const {
  for (const auto& stop_consumed_regst_infos : stop_consumed_regst_infos_) {
    Handler(stop_consumed_regst_infos);
  }
}

void ChainActGraph::ForEachRegstDescConsumerPathMeanDuration(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  double t1 = GetCurTime();
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
  double t2 = GetCurTime();
  LOG(INFO) << " total Time = " << t2 - t1;
}

void ChainActGraph::ForEachRegstDescConsumerPathIIScale(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  std::map<std::pair<int64_t, int64_t>, uint64_t> regst_desc_id_consumed2used_cnt;
  std::map<int64_t, uint64_t> regst_desc_id2produced_cnt;
  uint64_t max_cnt = 0;
  ForEachNode([&](const ChainActNode* node) {
    node->ForEachStartConsumerRegstInfos([&](const ConsumerRegstInfos* consumer_regst_infos) {
      for (const auto& pair : consumer_regst_infos->regst_id7producer) {
        int64_t produced_cnt = ++regst_desc_id2produced_cnt[pair.first];
        if (max_cnt < produced_cnt) { max_cnt = produced_cnt; }
        std::pair<int64_t, int64_t> regst_desc_id_consumed(
            pair.first, consumer_regst_infos->consumer->actor_id());
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
    if (IsActorWithConsumer(act_event)) {
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
  HashSet<std::shared_ptr<ConsumerRegstInfosCtx>> ctx_window;
  HashMap<const ConsumerRegstInfos*, std::shared_ptr<ConsumerRegstInfosCtx>>
      consumer_regst_infos2ctx;
  int node_cnt = 0;
  int iter = 0;
  ReverseDfsTopoForEachChainActNode([&](const ChainActNode* cur_node) {
    ++node_cnt;
    for (auto& ctx : ctx_window) {
      ++iter;
      CalcRegstActNodePathDuration(ctx.get(), cur_node);
    }
    cur_node->ForEachStartConsumerRegstInfos([&](const ConsumerRegstInfos* consumer_regst_infos) {
      const ChainActNode* consumer_node = cur_node;
      std::shared_ptr<ConsumerRegstInfosCtx> ctx(
          new ConsumerRegstInfosCtx(consumer_regst_infos, consumer_node));
      CHECK(ctx_window.emplace(ctx).second);
      CHECK(consumer_regst_infos2ctx.emplace(consumer_regst_infos, ctx).second);
    });
    cur_node->ForEachStopConsumerRegstInfos([&](const ConsumerRegstInfos* consumer_regst_infos) {
      ForEachRegstDescDuration(*(consumer_regst_infos2ctx.at(consumer_regst_infos)), Handler);
      ctx_window.erase(consumer_regst_infos2ctx.at(consumer_regst_infos));
      consumer_regst_infos2ctx.erase(consumer_regst_infos);
    });
    LOG(INFO) << "ctx_window.size = " << ctx_window.size();
  });
  LOG(INFO) << "node_cnt = " << node_cnt << " iter = " << iter;
}

void ChainActGraph::CalcRegstActNodePathDuration(ConsumerRegstInfosCtx* ctx,
                                                 const ChainActNode* node) const {
  double duration = 0;
  node->ForEachOutEdge([&](const ChainActEdge* out_edge) {
    const ChainActNode* out_node = out_edge->dst_node();
    if (ctx->node2duration_to_consumer.find(out_node) != ctx->node2duration_to_consumer.end()) {
      duration =
          std::max(duration, out_edge->duration() + ctx->node2duration_to_consumer.at(out_node));
    }
  });
  if (duration > 0) { ctx->node2duration_to_consumer[node] = duration; }
}

void ChainActGraph::ForEachRegstDescDuration(
    const ConsumerRegstInfosCtx& ctx,
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  // auto TopoOrderValue4Node = MakeGetterTopoOrderValue4Node();
  const ActEvent* consumer = ctx.consumer_regst_infos->consumer;
  for (auto const& pair : ctx.consumer_regst_infos->regst_id7producer) {
    const ActEvent* producer = pair.second;
    const ChainActNode* producer_node = Node4ActEvent(producer);
    double max_path_duration = 0;
    /*producer_node->ForEachOutEdge([&](const ChainActEdge* out_edge) {
      const ChainActNode* producer_out = out_edge->dst_node();
      if (out_edge->duration() >= pair.second->stop_time()
          && TopoOrderValue4Node(producer_out) >= TopoOrderValue4Node(Node4ActEvent(consumer))) {
        max_path_duration = std::max(max_path_duration,
            out_edge->duration() + ctx.node2duration_to_consumer.at(producer_out));
      }
    });*/
    double duration = ctx.node2duration_to_consumer.at(producer_node) + consumer->stop_time()
                      - producer->start_time();
    Handler(pair.first, consumer->actor_id(), duration);
  }
}

void ChainActGraph::InitTaskId2TaskProto() {
  for (const auto& task_proto : plan_->task()) {
    CHECK(task_id2task_proto_.emplace(task_proto.task_id(), &task_proto).second);
  }
}

void ChainActGraph::InitNodes(
    std::list<std::unique_ptr<ActEvent>>&& act_events,
    HashMap<std::pair<int64_t, int64_t>, const ActEvent*>* regst_uid2producer) {
  HashMap<std::pair<int64_t, int64_t>, std::list<std::unique_ptr<ActEvent>>>
      chain_id_with_act_id2act_events;
  for (auto& act_event : act_events) {
    int64_t act_id = act_event->act_id();
    const TaskProto& task_proto = GetTaskProto(act_event->actor_id());
    int64_t chain_id = task_proto.task_set_info().chain_id();
    std::pair<int64_t, int64_t> chain_act_id_pair(chain_id, act_id);
    chain_id_with_act_id2act_events[chain_act_id_pair].push_back(std::move(act_event));
  }
  for (auto& pair : chain_id_with_act_id2act_events) {
    ChainActNode* chain_act_node = new ChainActNode(std::move(pair.second));
    AddAllocatedNode(chain_act_node);
    chain_act_node->ForEachActEvent([&](const ActEvent* act_event) {
      int64_t act_id = act_event->act_id();
      const TaskProto& task_proto = GetTaskProto(act_event->actor_id());
      CHECK(act_event2chain_node_.emplace(act_event, chain_act_node).second);
      for (const auto& produced_regst_desc : task_proto.produced_regst_desc()) {
        int64_t regst_desc_id = produced_regst_desc.second.regst_desc_id();
        std::pair<int64_t, int64_t> regst_uid(regst_desc_id, act_id);
        CHECK(regst_uid2producer->emplace(regst_uid, act_event).second);
      }
    });
  }
}

void ChainActGraph::InitEdges(
    const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer,
    HashMap<const ActEvent*, std::list<std::pair<int64_t, const ActEvent*>>>*
        consumer2regst_id7producer) {
  ForEachNode([&](const ChainActNode* node) {
    HashMap<ChainActNode*, double> producer_node2max_stop_time;
    ChainActNode* mut_node = nullptr;
    node->ForEachActEvent([&](const ActEvent* consumer) {
      mut_node = act_event2chain_node_.at(consumer);
      for (const auto& readable : consumer->readable_regst_infos()) {
        std::pair<int64_t, int64_t> regst_uid(readable.regst_desc_id(), readable.act_id());
        const auto& producer_it = regst_uid2producer.find(regst_uid);
        if (producer_it == regst_uid2producer.end()) { continue; }
        actor_with_consumer_.emplace(producer_it->second);
        std::pair<int64_t, const ActEvent*> regst_id7producer(regst_uid.first, producer_it->second);
        (*consumer2regst_id7producer)[consumer].push_back(regst_id7producer);
        ChainActNode* producer_node = act_event2chain_node_.at(producer_it->second);
        if (producer_node == node) { continue; }
        double& max_stop_time = producer_node2max_stop_time[producer_node];
        max_stop_time = std::max(max_stop_time, producer_it->second->stop_time());
      }
    });
    CHECK_EQ(node, mut_node);
    for (const auto& pair : producer_node2max_stop_time) {
      ChainActEdge* edge = new ChainActEdge(pair.second);
      AddAllocatedEdge(edge);
      Connect(pair.first, edge, mut_node);
    }
  });
}

void ChainActGraph::InitNodeStartConsumerRegstInfos(
    const HashMap<const ActEvent*, std::list<std::pair<int64_t, const ActEvent*>>>&
        consumer2regst_id7producer) const {
  for (const auto& pair : consumer2regst_id7producer) {
    ChainActNode* consumer_node = act_event2chain_node_.at(pair.first);
    auto consumer_regst_infos = std::make_unique<ConsumerRegstInfos>(pair.first, pair.second);
    consumer_node->AddStartConsumerRegstInfos(std::move(consumer_regst_infos));
  }
}

void ChainActGraph::InitNodeStopConsumerRegstInfos() const {
  auto TopoOrderValue4Node = MakeGetterTopoOrderValue4Node();
  ForEachNode([&](const ChainActNode* node) {
    node->ForEachStartConsumerRegstInfos([&](const ConsumerRegstInfos* consumer_regst_infos) {
      int64_t max_producer_topo_order_value = -1;
      const ActEvent* last_producer = nullptr;
      for (const auto& pair : consumer_regst_infos->regst_id7producer) {
        const ActEvent* producer = pair.second;
        int64_t cur_producer_topo_order_value = TopoOrderValue4Node(Node4ActEvent(producer));
        if (cur_producer_topo_order_value > max_producer_topo_order_value) {
          max_producer_topo_order_value = cur_producer_topo_order_value;
          last_producer = producer;
        }
      }
      act_event2chain_node_.at(last_producer)->AddStopConsumerRegstInfos(consumer_regst_infos);
    });
  });
}

std::function<int64_t(const ChainActNode*)> ChainActGraph::MakeGetterTopoOrderValue4Node() const {
  auto node2topo_order_value = std::make_shared<HashMap<const ChainActNode*, int64_t>>();
  int64_t topo_order_value = -1;
  ReverseDfsTopoForEachChainActNode([&](const ChainActNode* chain_act_node) {
    (*node2topo_order_value)[chain_act_node] = ++topo_order_value;
  });
  return
      [node2topo_order_value](const ChainActNode* node) { return node2topo_order_value->at(node); };
}

const ChainActNode* ChainActGraph::Node4ActEvent(const ActEvent* act_event) const {
  return act_event2chain_node_.at(act_event);
}

void ChainActGraph::DfsTopoForEachChainActNode(
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

void ChainActGraph::ReverseDfsTopoForEachChainActNode(
    const std::function<void(const ChainActNode*)>& Handler) const {
  std::list<const ChainActNode*> topo_nodes;
  DfsTopoForEachChainActNode(
      [&](const ChainActNode* chain_act_node) { topo_nodes.push_back(chain_act_node); });
  for (auto iter = topo_nodes.rbegin(); iter != topo_nodes.rend(); ++iter) { Handler(*iter); }
}

void ChainActGraph::ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const {
  ForEachNode([&](const ChainActNode* node) { node->ForEachActEvent(Handler); });
}

bool ChainActGraph::IsActorWithConsumer(const ActEvent* act_event) const {
  return actor_with_consumer_.find(act_event) != actor_with_consumer_.end();
}

ChainActGraph::ChainActGraph(const Plan& plan, std::list<std::unique_ptr<ActEvent>>&& act_events)
    : plan_(&plan) {
  HashMap<std::pair<int64_t, int64_t>, const ActEvent*> regst_uid2producer_act_event;
  HashMap<const ActEvent*, std::list<std::pair<int64_t, const ActEvent*>>>
      consumer2regst_id7producer;
  InitTaskId2TaskProto();
  InitNodes(std::move(act_events), &regst_uid2producer_act_event);
  InitEdges(regst_uid2producer_act_event, &consumer2regst_id7producer);
  InitNodeStartConsumerRegstInfos(consumer2regst_id7producer);
  InitNodeStopConsumerRegstInfos();
  LOG(INFO) << "ChainActGraph Init Completed!";
}

}  // namespace oneflow
