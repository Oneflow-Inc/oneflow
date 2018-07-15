#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

namespace {
std::string GenRegstUid(int64_t regst_desc_id, int64_t producer_act_id) {
  return std::to_string(regst_desc_id) + ":" + std::to_string(producer_act_id);
}

int64_t RegstDescId4RegstUid(const std::string& regst_uid) {
  std::stringstream ss;
  ss << regst_uid;
  int64_t regst_desc_id = 0;
  ss >> regst_desc_id;
  return regst_desc_id;
}

}  // namespace

class DepthRangeChainActSubGraph final : public Graph<const ChainActNode, const ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DepthRangeChainActSubGraph);
  DepthRangeChainActSubGraph(const ChainActGraph* chain_act_graph, const Range& depth_range,
                             const std::list<std::string>& regst_uids);
  ~DepthRangeChainActSubGraph() = default;
  void CalTotalDuration(const std::function<void(std::string, int64_t, double)>& Handler) const;

 private:
  void TopoForEachChainActNode(const std::list<const ChainActNode*>& starts,
                               const std::function<void(const ChainActNode*)>& Handler) const;
  void ForEachInEdge(const ChainActNode* node,
                     const std::function<void(const ChainActEdge*)>& Handler) const;

  const ChainActGraph* graph_;
  Range depth_range_;
  std::list<std::string> regst_uids_;
};

DepthRangeChainActSubGraph::DepthRangeChainActSubGraph(const ChainActGraph* chain_act_graph,
                                                       const Range& depth_range,
                                                       const std::list<std::string>& regst_uids)
    : graph_(chain_act_graph), depth_range_(depth_range), regst_uids_(regst_uids) {}

void DepthRangeChainActSubGraph::ForEachInEdge(
    const ChainActNode* node, const std::function<void(const ChainActEdge*)>& Handler) const {
  for (const ChainActEdge* in_edge : node->in_edges()) {
    const ChainActNode* in_node = in_edge->src_node();
    if (in_node->depth() >= depth_range_.begin() && in_node->depth() <= depth_range_.end()) {
      Handler(in_edge);
    }
  }
}

void DepthRangeChainActSubGraph::TopoForEachChainActNode(
    const std::list<const ChainActNode*>& starts,
    const std::function<void(const ChainActNode*)>& Handler) const {
  for (int64_t depth = depth_range_.begin(); depth <= depth_range_.end(); ++depth) {
    for (const ChainActNode* node : graph_->Nodes4Depth(depth)) { Handler(node); }
  }
}

void DepthRangeChainActSubGraph::CalTotalDuration(
    const std::function<void(std::string, int64_t, double)>& Handler) const {
  HashSet<const ChainActNode*> total_producers;
  HashSet<const ChainActNode*> regst_duration_window_start_nodes;
  HashMap<int64_t, HashSet<const ChainActNode*>> end_node2start_nodes;
  HashMap<const ChainActNode*, HashMap<const ChainActNode*, double>> src2dst_duration;
  const auto& starts = graph_->Nodes4Depth(depth_range_.begin());
  for (const auto& regst_uid : regst_uids_) {
    const ChainActNode* producer = graph_->ProducerNode4RegstUid(regst_uid);
    if (total_producers.find(producer) != total_producers.end()) { continue; }
    total_producers.insert(producer);
    int64_t max_topo_id = -1;
    for (const ChainActNode* consumer : graph_->AllConsumers4Producer(producer)) {
      max_topo_id = std::max(max_topo_id, consumer->topo_id());
    }
    end_node2start_nodes[max_topo_id].insert(producer);
  }
  auto CalDuration = [&](const ChainActNode* src, const ChainActNode* dst) {
    double duration = 0;
    ForEachInEdge(dst, [&](const ChainActEdge* in_edge) {
      const ChainActNode* in_node = in_edge->src_node();
      if (src2dst_duration[src].find(in_node) != src2dst_duration[src].end()) {
        duration = std::max(duration, in_edge->duration() + src2dst_duration[src][in_node]);
      }
    });
    if (duration != 0) { src2dst_duration[src][dst] = duration; }
  };
  TopoForEachChainActNode(starts, [&](const ChainActNode* cur_node) {
    for (const ChainActNode* pre_node : regst_duration_window_start_nodes) {
      CalDuration(pre_node, cur_node);
    }
    if (total_producers.find(cur_node) != total_producers.end()) {
      src2dst_duration[cur_node][cur_node] = 0;
      regst_duration_window_start_nodes.insert(cur_node);
    }
    for (const ChainActNode* start_node : end_node2start_nodes[cur_node->topo_id()]) {
      regst_duration_window_start_nodes.erase(start_node);
    }
  });
  for (const auto& regst_uid : regst_uids_) {
    const ActEvent* producer_act_event = graph_->ProducerActEvent4RegstUid(regst_uid);
    const ChainActNode* producer = graph_->Node4ActEvent(producer_act_event);
    for (const ActEvent* consumer_act_event : graph_->ConsumerActEvents4RegstUid(regst_uid)) {
      const ChainActNode* consumer = graph_->Node4ActEvent(consumer_act_event);
      double duration = src2dst_duration[producer][consumer] + consumer_act_event->stop_time()
                        - producer_act_event->start_time();
      Handler(regst_uid, consumer_act_event->actor_id(), duration);
    }
  }
}

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
  ForEachRegstUidConsumerPathDuration(
      [&](const std::string& regst_uid, int64_t consumer_actor_id, double duration) {
        int64_t regst_desc_id = RegstDescId4RegstUid(regst_uid);
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
  for (const auto& pair : regst_uid2consumer_act_events_) {
    int64_t regst_desc_id = RegstDescId4RegstUid(pair.first);
    int64_t produced_cnt = ++regst_desc_id2produced_cnt[regst_desc_id];
    if (max_cnt < produced_cnt) { max_cnt = produced_cnt; }
    for (const ActEvent* act_event : pair.second) {
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
      act_event2chain_node_.insert({act_event, chain_act_node});
      act_event2has_consumer_[act_event] = false;
      for (const auto& pair : task_proto.produced_regst_desc()) {
        int64_t regst_desc_id = pair.second.regst_desc_id();
        const auto& regst_uid = GenRegstUid(regst_desc_id, act_id);
        regst_uid2producer_act_event_.insert({regst_uid, act_event});
      }
    }
  }
}

void ChainActGraph::InitEdges() {
  ForEachNode([&](ChainActNode* node) {
    HashMap<ChainActNode*, double> producer_node2max_stop_time;
    for (const ActEvent* act_event : node->act_events()) {
      for (const auto& readable : act_event->readable_regst_infos()) {
        const auto& regst_uid = GenRegstUid(readable.regst_desc_id(), readable.act_id());
        const auto& producer_act_event_it = regst_uid2producer_act_event_.find(regst_uid);
        if (producer_act_event_it == regst_uid2producer_act_event_.end()) { continue; }
        act_event2has_consumer_[producer_act_event_it->second] = true;
        regst_uid2consumer_act_events_[regst_uid].push_back(act_event);
        ChainActNode* producer_node = ProducerNode4RegstUid(regst_uid);
        producer2consumers_[producer_node].push_back(node);
        if (producer_node == node) { continue; }
        double& max_stop_time = producer_node2max_stop_time[producer_node];
        max_stop_time = std::max(max_stop_time, producer_act_event_it->second->stop_time());
      }
    }
    for (const auto& pair : producer_node2max_stop_time) {
      ChainActEdge* edge = NewEdge();
      Connect(pair.first, edge, node);
      edge->set_duration(pair.second);
    }
  });
}

void ChainActGraph::InitDepth7TopoId() {
  std::list<ChainActNode*> sources;
  ForEachNode([&](ChainActNode* node) {
    if (node->in_edges().empty()) { sources.push_back(node); }
  });
  int64_t topo_id = -1;
  int64_t max_depth = -1;
  TopoForEachChainActNode(sources, [&](ChainActNode* chain_act_node) {
    int64_t depth = -1;
    chain_act_node->ForEachNodeOnInEdge(
        [&](ChainActNode* in_node) { depth = std::max(depth, in_node->depth()); });
    if (depth == -1) { depth = max_depth; }
    ++depth;
    ++topo_id;
    chain_act_node->set_depth(depth);
    chain_act_node->set_topo_id(topo_id);
    depth2nodes_[depth].push_back(chain_act_node);
    max_depth = std::max(max_depth, depth);
  });
}

void ChainActGraph::ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const {
  for (const ActEvent& act_event : *act_events_) { Handler(&act_event); }
}

void ChainActGraph::ForEachDepthRangeRegstUids(
    const std::function<void(const Range& range, const std::list<std::string>& regst_uids)>&
        Handler) const {
  struct CompareRangeSize {
    bool operator()(const Range& lhs, const Range& rhs) {
      return lhs.size() > rhs.size() || (lhs.size() == rhs.size() && lhs.begin() < rhs.begin());
    }
  };
  struct CompareRange {
    bool operator()(const Range& lhs, const Range& rhs) {
      if (lhs.begin() == rhs.begin() || lhs.end() == rhs.end()) { return false; }
      return lhs.begin() <= rhs.begin() && lhs.end() <= rhs.end();
    }
  };
  std::map<Range, std::list<std::string>, CompareRangeSize> depth_range2regst_uids;
  std::map<Range, std::list<std::string>, CompareRange> depth_range_cluster2regst_uids;
  for (const auto& pair : regst_uid2producer_act_event_) {
    const auto& consumer_act_event_it = regst_uid2consumer_act_events_.find(pair.first);
    if (consumer_act_event_it == regst_uid2consumer_act_events_.end()) { continue; }
    int64_t begin = ProducerNode4RegstUid(pair.first)->depth();
    int64_t end = 0;
    for (const ActEvent* consumer_act_event : consumer_act_event_it->second) {
      end = std::max(end, Node4ActEvent(consumer_act_event)->depth());
    }
    depth_range2regst_uids[Range(begin, end)].push_back(pair.first);
  }
  for (const auto& pair : depth_range2regst_uids) {
    auto iter = depth_range_cluster2regst_uids.find(pair.first);
    if (iter != depth_range_cluster2regst_uids.end()) {
      for (const auto& regst_uid : pair.second) { iter->second.push_back(regst_uid); }
    } else {
      for (const auto& regst_uid : pair.second) {
        depth_range_cluster2regst_uids[pair.first].push_back(regst_uid);
      }
    }
  }
  for (const auto& pair : depth_range_cluster2regst_uids) { Handler(pair.first, pair.second); }
}

void ChainActGraph::ForEachRegstUidConsumerPathDuration(
    const std::function<void(const std::string&, int64_t, double)>& Handler) const {
  ForEachDepthRangeRegstUids([&](const Range& range, const std::list<std::string>& regst_uids) {
    DepthRangeChainActSubGraph depth_range_subgraph(this, range, regst_uids);
    depth_range_subgraph.CalTotalDuration(
        [&](std::string regst_uid, int64_t consumer_actor_id, double duration) {
          Handler(regst_uid, consumer_actor_id, duration);
        });
  });
}

ChainActGraph::ChainActGraph(const Plan& plan, std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  InitTaskId2TaskProto();
  InitNodes();
  InitEdges();
  InitDepth7TopoId();
}

}  // namespace oneflow
