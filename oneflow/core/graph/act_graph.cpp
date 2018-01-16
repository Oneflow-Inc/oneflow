#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/graph/visitor.h"

namespace oneflow {

namespace {

std::string GenRegstUid(int64_t regst_desc_id, int64_t producer_act_id) {
  return std::to_string(regst_desc_id) + ":" + std::to_string(producer_act_id);
}

void ForEachConnectedNode(const ActNode* node,
                          const std::function<void(const ActNode*)>& Handler) {
  node->ForEachNodeOnInEdge(Handler);
  node->ForEachNodeOnOutEdge(Handler);
}

void BfsForEachActNode(const std::list<const ActNode*>& starts,
                       const std::function<void(const ActNode*)>& Handler) {
  Visitor<const ActNode*>::BfsForEach(starts, ForEachConnectedNode, Handler);
}

void TopoForEachActNode(const std::list<const ActNode*>& starts,
                        const std::function<void(const ActNode*)>& Handler) {
  auto ForEachInNode = std::bind(&ActNode::ForEachNodeOnInEdge,
                                 std::placeholders::_1, std::placeholders::_2);
  auto ForEachOutNode = std::bind(&ActNode::ForEachNodeOnOutEdge,
                                  std::placeholders::_1, std::placeholders::_2);
  Visitor<const ActNode*>::TopoForEach(starts, ForEachInNode, ForEachOutNode,
                                       Handler);
}

void InitSubGraphAncestors(
    const std::list<const ActNode*>& sources,
    HashMap<const ActNode*, std::unordered_set<const ActNode*>>*
        node2ancestors) {
  TopoForEachActNode(sources, [&](const ActNode* node) {
    node->ForEachNodeOnInEdge([&](const ActNode* prev) {
      (*node2ancestors)[node].insert((*node2ancestors)[prev].begin(),
                                     (*node2ancestors)[prev].end());
      (*node2ancestors)[node].insert(prev);
    });
  });
}

double CalcLongestPathTime(
    const ActNode* src, const ActNode* dst,
    const std::function<bool(const ActNode* src, const ActNode* dst)>&
        IsReachable) {
  using ForEachNodeFn = std::function<void(const ActNode*)>;
  auto ForEachInNode = [&](const ActNode* node, const ForEachNodeFn& Handler) {
    node->ForEachNodeOnInEdge([&](const ActNode* in) {
      if (in == src || IsReachable(src, in)) { Handler(in); }
    });
  };
  auto ForEachOutNode = [&](const ActNode* node, const ForEachNodeFn& Handler) {
    node->ForEachNodeOnOutEdge([&](const ActNode* out) {
      if (out == dst || IsReachable(out, dst)) { Handler(out); }
    });
  };
  HashMap<const ActNode*, double> node2longest_path_time;
  auto SetLongestPathTime = [&](const ActNode* node) {
    double time = 0;
    ForEachInNode(node, [&](const ActNode* in_node) {
      time = std::max(time, in_node->time());
    });
    node2longest_path_time[node] = time + node->time();
  };
  Visitor<const ActNode*>::TopoForEach({src}, ForEachInNode, ForEachOutNode,
                                       SetLongestPathTime);
  return node2longest_path_time.at(dst);
}

double CalcLongestPathTime(
    const ActNode* src, const std::list<const ActNode*>& dst_nodes,
    const std::function<bool(const ActNode* src, const ActNode* dst)>&
        IsReachable) {
  double time = 0;
  for (const ActNode* dst : dst_nodes) {
    time = std::max(time, CalcLongestPathTime(src, dst, IsReachable));
  }
  return time;
}

void ForEachConnectedSubGraphSources(
    const std::list<const ActNode*>& sources,
    const std::function<void(const std::list<const ActNode*>&)>& Handler) {
  HashMap<const ActNode*, bool> visited;
  for (const ActNode* source : sources) {
    if (visited[source]) { continue; }
    std::list<const ActNode*> sub_graph_sources;
    BfsForEachActNode({source}, [&](const ActNode* node) {
      if (node->in_edges().empty()) { sub_graph_sources.push_back(node); }
    });
    Handler(sub_graph_sources);
    for (const ActNode* sub_graph_source : sub_graph_sources) {
      visited[sub_graph_source] = true;
    }
  }
}

}  // namespace

void ActGraph::ForEachRegstDescLifeTime(
    const std::function<void(int64_t, double)>& Handler) const {
  HashMap<int64_t, double> regst_desc_id2total_time;
  HashMap<int64_t, int> regst_desc_id2cnt;
  ForEachRegstUidLifeTime([&](double time, int64_t regst_desc_id, int64_t) {
    regst_desc_id2total_time[regst_desc_id] += time;
    ++regst_desc_id2cnt[regst_desc_id];
  });
  for (const auto& pair : regst_desc_id2total_time) {
    Handler(pair.first, pair.second / regst_desc_id2cnt.at(pair.first));
  }
}

void ActGraph::ForEachSubGraphRegstUidLifeTime(
    const std::list<const ActNode*>& sources,
    const std::function<void(double, int64_t, int64_t)>& Handler) const {
  HashMap<const ActNode*, std::unordered_set<const ActNode*>> node2ancestors;
  TopoForEachActNode(sources, [&](const ActNode* node) {
    node->ForEachNodeOnInEdge([&](const ActNode* prev) {
      node2ancestors[node].insert(node2ancestors[prev].begin(),
                                  node2ancestors[prev].end());
      node2ancestors[node].insert(prev);
    });
  });
  auto IsReachable = [&](const ActNode* src, const ActNode* dst) {
    const auto& it = node2ancestors.find(dst);
    if (it == node2ancestors.end()) { return false; }
    return it->second.find(src) != it->second.end();
  };
  BfsForEachActNode(sources, [&](const ActNode* node) {
    int64_t actor_id = node->actor_id();
    for (int64_t regst_desc_id : producer_id2regst_desc_ids_.at(actor_id)) {
      const auto& regst_uid = GenRegstUid(regst_desc_id, node->act_id());
      const auto& consumers_it = regst_uid2consumer_acts_.find(regst_uid);
      if (consumers_it == regst_uid2consumer_acts_.end()) { continue; }
      const auto& consumers = consumers_it->second;
      if (consumers.empty()) { continue; }
      double time = CalcLongestPathTime(node, consumers, IsReachable);
      Handler(time, regst_desc_id, node->act_id());
    }
  });
}

void ActGraph::ForEachRegstUidLifeTime(
    const std::function<void(double, int64_t, int64_t)>& Handler) const {
  ForEachConnectedSubGraphSources(
      sources_, [&](const std::list<const ActNode*>& sources) {
        ForEachSubGraphRegstUidLifeTime(sources, Handler);
      });
}

void ActGraph::InitNodes() {
  for (const ActEvent& act_event : *act_events_) {
    int64_t actor_id = act_event.actor_id();
    int64_t act_id = act_event.act_id();
    ActNode* act_node = new ActNode(&act_event);
    AddAllocatedNode(act_node);
    for (int64_t regst_desc_id : producer_id2regst_desc_ids_.at(actor_id)) {
      regst_desc_id2producer_act_ids_[regst_desc_id].push_back(act_id);
      const auto& regst_uid = GenRegstUid(regst_desc_id, act_id);
      regst_uid2producer_node_.insert({regst_uid, act_node});
    }
  }
}

void ActGraph::InitEdges() {
  ForEachNode([&](ActNode* node) {
    for (const auto& readable : node->act_event().readable_regst_infos()) {
      const auto& regst_uid =
          GenRegstUid(readable.regst_desc_id(), readable.act_id());
      ActNode* producer = regst_uid2producer_node_.at(regst_uid);
      Connect(producer, NewEdge(), node);
      regst_uid2consumer_acts_[regst_uid].push_back(node);
    }
  });
}

void ActGraph::InitProducerId2RegstDescIds() {
  for (const TaskProto& task : plan().task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      producer_id2regst_desc_ids_[pair.second.regst_desc_id()].push_back(
          pair.second.producer_task_id());
    }
  }
}

void ActGraph::InitSources() {
  ForEachNode([&](ActNode* node) {
    if (node->in_edges().empty()) { sources_.push_back(node); }
  });
}

ActGraph::ActGraph(const Plan& plan,
                   std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  InitProducerId2RegstDescIds();
  InitNodes();
  InitEdges();
  InitSources();
}

}  // namespace oneflow
