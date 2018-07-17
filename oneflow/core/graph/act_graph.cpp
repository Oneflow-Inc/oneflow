#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

namespace {
std::string GenRegstUid(int64_t regst_desc_id, int64_t producer_act_id) {
  return std::to_string(regst_desc_id) + ":" + std::to_string(producer_act_id);
}

}  // namespace

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
  double t1 = GetCurTime();
  std::map<std::pair<int64_t, int64_t>, double> regst_desc_id_consumed2duration;
  std::map<std::pair<int64_t, int64_t>, int> regst_desc_id_consumed2cnt;
  HashSet<RegstAct*> regst_act_window;
  auto CalDuration = [&](const ChainActNode* node, RegstAct* regst_act) {
    double duration = 0;
    ForEachInEdge(node, [&](const ChainActEdge* in_edge) {
      const ChainActNode* in_node = in_edge->src_node();
      if (regst_act->node2producer_duration.find(in_node)
          != regst_act->node2producer_duration.end()) {
        duration =
            std::max(duration, in_edge->duration() + regst_act->node2producer_duration[in_node]);
      }
    });
    if (duration > 0) { regst_act->node2producer_duration[node] = duration; }
  };
  for (int64_t i = 0; i < topo_nodes_.size(); ++i) {
    const ChainActNode* cur_node = topo_nodes_.at(i);
    for (RegstAct* regst_act : regst_act_window) {
      if (regst_act->fake_producer_outs.find(cur_node) != regst_act->fake_producer_outs.end()) {
        continue;
      }
      CalDuration(cur_node, regst_act);
    }
    for (RegstAct* regst_act : cur_node->produced_regst_acts()) {
      regst_act->node2producer_duration[cur_node] = 0;
      for (const ChainActEdge* out_edge : cur_node->out_edges()) {
        if (out_edge->duration() < regst_act->producer_act_event->stop_time()) {
          regst_act->fake_producer_outs.insert(out_edge->dst_node());
        }
      }
      regst_act_window.insert(regst_act);
    }
    for (RegstAct* regst_act : cur_node->last_consumed_regst_acts()) {
      for (const ActEvent* consumer_act_event : regst_act->consumer_act_events) {
        int64_t regst_desc_id = regst_act->regst_desc_id;
        int64_t consumer_actor_id = consumer_act_event->actor_id();
        double duration = regst_act->node2producer_duration[Node4ActEvent(consumer_act_event)]
                          + consumer_act_event->stop_time()
                          - regst_act->producer_act_event->start_time();
        std::string regst_uid = GenRegstUid(regst_desc_id, regst_act->act_id);
        LOG(INFO) << "regst_uid = " << regst_uid << " chain_duration = " << duration;
        std::pair<int64_t, int64_t> regst_desc_id_consumed(regst_desc_id, consumer_actor_id);
        regst_desc_id_consumed2duration[regst_desc_id_consumed] += duration;
        ++regst_desc_id_consumed2cnt[regst_desc_id_consumed];
      }
      regst_act_window.erase(regst_act);
      // delete this regst_act
    }
  }
  double t2 = GetCurTime();
  LOG(INFO) << "total time = " << t2 - t1;
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
  HashMap<std::string, std::list<ActEvent*>> node_str2act_events;
  for (ActEvent& act_event : *act_events_) {
    int64_t actor_id = act_event.actor_id();
    int64_t act_id = act_event.act_id();
    const TaskProto& task_proto = GetTaskProto(actor_id);
    int64_t chain_id = task_proto.task_set_info().chain_id();
    node_str2act_events[std::to_string(chain_id) + ":" + std::to_string(act_id)].push_back(
        &act_event);
  }
  for (auto node_str : node_str2act_events) {
    ChainActNode* chain_act_node = new ChainActNode(node_str.second);
    AddAllocatedNode(chain_act_node);
    for (const ActEvent* act_event : chain_act_node->act_events()) {
      int64_t actor_id = act_event->actor_id();
      int64_t act_id = act_event->act_id();
      const TaskProto& task_proto = GetTaskProto(actor_id);
      act_event2chain_node_.emplace(act_event, chain_act_node);
      act_event2has_consumer_[act_event] = false;
      for (const auto& pair : task_proto.produced_regst_desc()) {
        int64_t regst_desc_id = pair.second.regst_desc_id();
        std::string regst_uid = GenRegstUid(regst_desc_id, act_id);
        RegstAct* regst_act = new RegstAct;
        regst_act->regst_desc_id = regst_desc_id;
        regst_act->act_id = act_id;
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
        std::string regst_uid = GenRegstUid(readable.regst_desc_id(), readable.act_id());
        const auto& regst_act_it = regst_uid2regst_act_.find(regst_uid);
        if (regst_act_it == regst_uid2regst_act_.end()) { continue; }
        act_event2has_consumer_[regst_act_it->second->producer_act_event] = true;
        regst_act_it->second->consumer_act_events.push_back(act_event);
        ChainActNode* producer_node = Node4ActEvent(regst_act_it->second->producer_act_event);
        producer_node->set_produced_regst_acts(regst_act_it->second);
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

void ChainActGraph::InitOrderInGraph() {
  std::list<ChainActNode*> sources;
  ForEachNode([&](ChainActNode* node) {
    if (node->in_edges().empty()) { sources.push_back(node); }
  });
  int64_t order_in_graph = -1;
  TopoForEachChainActNode(sources, [&](ChainActNode* chain_act_node) {
    chain_act_node->set_order_in_graph(++order_in_graph);
    topo_nodes_.push_back(chain_act_node);
  });
  for (auto& pair : regst_uid2regst_act_) {
    RegstAct* regst_act = pair.second;
    int64_t last_consumer_order_in_graph = -1;
    for (const ActEvent* consumer_act_event : regst_act->consumer_act_events) {
      last_consumer_order_in_graph = std::max(last_consumer_order_in_graph,
                                              Node4ActEvent(consumer_act_event)->order_in_graph());
    }
    if (last_consumer_order_in_graph != -1) {
      topo_nodes_.at(last_consumer_order_in_graph)->set_last_consumed_regst_acts(regst_act);
    }
  }
}

void ChainActGraph::ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const {
  for (const ActEvent& act_event : *act_events_) { Handler(&act_event); }
}

void ChainActGraph::ForEachInEdge(const ChainActNode* node,
                                  const std::function<void(const ChainActEdge*)>& Handler) const {
  for (const ChainActEdge* in_edge : node->in_edges()) { Handler(in_edge); }
}

ChainActGraph::ChainActGraph(const Plan& plan, std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  InitTaskId2TaskProto();
  InitNodes();
  InitEdges();
  InitOrderInGraph();
}

}  // namespace oneflow
