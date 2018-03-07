#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

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

class RegstActSubGraph final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstActSubGraph);
  RegstActSubGraph(const std::string& regst_uid, const ActNode* producer_node,
                   const HashSet<const ActNode*>& partial_producer_outs,
                   const HashSet<const ActNode*>& partial_consumer_nodes,
                   const std::list<const ActNode*>& fake_sources_super_set);
  ~RegstActSubGraph() = default;

  double CalcLongestPathDuration() const;

  // Getters
  const std::string& regst_uid() const { return regst_uid_; }

 private:
  void TopoForEachActNode(
      const std::function<void(const ActNode*)>& Handler) const;
  void ForEachInNode(const ActNode* node,
                     const std::function<void(const ActNode*)>& Handler) const;
  void ForEachOutNode(const ActNode* node,
                      const std::function<void(const ActNode*)>& Handler) const;
  std::list<const ActNode*> CalcSources() const;
  bool IsSource(const ActNode* node) const;

  std::string regst_uid_;
  const ActNode* producer_node_;
  HashSet<const ActNode*> partial_producer_outs_;
  HashSet<const ActNode*> partial_consumer_nodes_;
  HashSet<const ActNode*> fake_sources_;
  int64_t max_depth_;
};

RegstActSubGraph::RegstActSubGraph(
    const std::string& regst_uid, const ActNode* producer_node,
    const HashSet<const ActNode*>& partial_producer_outs,
    const HashSet<const ActNode*>& partial_consumer_nodes,
    const std::list<const ActNode*>& fake_sources_super_set)
    : regst_uid_(regst_uid),
      producer_node_(producer_node),
      partial_producer_outs_(partial_producer_outs),
      partial_consumer_nodes_(partial_consumer_nodes) {
  for (const ActNode* node : fake_sources_super_set) {
    if (partial_producer_outs_.find(node) == partial_producer_outs_.end()) {
      fake_sources_.insert(node);
    }
  }
  max_depth_ = 0;
  for (const ActNode* node : partial_consumer_nodes_) {
    max_depth_ = std::max(max_depth_, node->depth());
  }
}

std::list<const ActNode*> RegstActSubGraph::CalcSources() const {
  std::list<const ActNode*> sources;
  for (const ActNode* node : fake_sources_) { sources.push_back(node); }
  sources.push_back(producer_node_);
  return sources;
}

bool RegstActSubGraph::IsSource(const ActNode* node) const {
  return node == producer_node_
         || fake_sources_.find(node) != fake_sources_.end();
}

void RegstActSubGraph::TopoForEachActNode(
    const std::function<void(const ActNode*)>& Handler) const {
  TODO();
}

void RegstActSubGraph::ForEachInNode(
    const ActNode* node,
    const std::function<void(const ActNode*)>& Handler) const {
  if (IsSource(node)) { return; }
  node->ForEachNodeOnInEdge(Handler);
}

void RegstActSubGraph::ForEachOutNode(
    const ActNode* node,
    const std::function<void(const ActNode*)>& Handler) const {
  if (node->depth() >= max_depth_) { return; }
  node->ForEachNodeOnOutEdge(Handler);
}

double RegstActSubGraph::CalcLongestPathDuration() const {
  auto Duration4Node = [&](const ActNode* node) {
    if (fake_sources_.find(node) != fake_sources_.end()) {
      return std::numeric_limits<double>::min();
    } else {
      return node->Duration();
    }
  };
  HashMap<const ActNode*, double> node2longest_path_duration;
  TopoForEachActNode([&](const ActNode* node) {
    double duration = 0;
    ForEachInNode(node, [&](const ActNode* in_node) {
      duration = std::max(duration, node2longest_path_duration[in_node]);
    });
    node2longest_path_duration[node] = duration + Duration4Node(node);
  });
  double duration = 0;
  for (const ActNode* node : partial_consumer_nodes_) {
    duration = std::max(duration, node2longest_path_duration.at(node));
  }
  return duration;
}

void ActNode::AddConsumerNode(const std::string& regst_uid,
                              const ActNode* consumer_node) {
  regst_uid2consumer_nodes_[regst_uid].push_back(consumer_node);
}

std::string ActNode::VisualStr() const {
  std::string name = std::to_string(act_id()) + "\\n";
  name += std::to_string(task_proto_->task_id()) + "\\n";
  for (const auto& exec_node : task_proto_->exec_sequence().exec_node()) {
    name = name + exec_node.kernel_conf().op_conf().name() + "\\n";
  }
  return name;
}

void ActNode::ForEachProducedRegstDescId(
    const std::function<void(int64_t)>& Handler) const {
  for (const auto& pair : task_proto_->produced_regst_desc()) {
    Handler(pair.second.regst_desc_id());
  }
}

void ActGraph::ForEachRegstDescIIScale(
    const std::function<void(int64_t, double)>& Handler) const {
  HashMap<int64_t, size_t> regst_desc_id2used_cnt;
  size_t max_used_cnt = 0;
  for (const auto& pair : regst_uid2consumer_nodes_) {
    int64_t regst_desc_id = RegstDescId4RegstUid(pair.first);
    int64_t used_cnt = ++regst_desc_id2used_cnt[regst_desc_id];
    if (max_used_cnt < used_cnt) { max_used_cnt = used_cnt; }
  }
  for (const auto& pair : regst_desc_id2used_cnt) {
    Handler(pair.first, max_used_cnt / static_cast<double>(pair.second));
  }
}

void ActGraph::ForEachRegstDescMeanDuration(
    const std::function<void(int64_t, double)>& Handler) const {
  HashMap<int64_t, double> regst_desc_id2duration;
  HashMap<int64_t, int> regst_desc_id2cnt;
  ForEachRegstUidDuration([&](const std::string& regst_uid, double duration) {
    int64_t regst_desc_id = RegstDescId4RegstUid(regst_uid);
    regst_desc_id2duration[regst_desc_id] += duration;
    ++regst_desc_id2cnt[regst_desc_id];
  });
  for (const auto& pair : regst_desc_id2duration) {
    Handler(pair.first, pair.second / regst_desc_id2cnt.at(pair.first));
  }
}

void ActGraph::ForEachRegstActSubGraph(
    const std::function<void(const RegstActSubGraph&)>& Handler) const {
  ForEachDepthRangeRegstUids(
      [&](const Range& range, const std::list<std::string>& regst_uids) {
        TODO();
      });
}

void ActGraph::ForEachRegstUidDuration(
    const std::function<void(const std::string& regst_uid, double duration)>&
        Handler) const {
  HashMap<std::string, double> regst_uid2duration;
  ForEachRegstActSubGraph([&](const RegstActSubGraph& regst_csm_graph) {
    const std::string regst_uid = regst_csm_graph.regst_uid();
    double duration = regst_csm_graph.CalcLongestPathDuration();
    regst_uid2duration[regst_uid] =
        std::max(regst_uid2duration[regst_uid], duration);
  });
  for (const auto& pair : regst_uid2duration) {
    Handler(pair.first, pair.second);
  }
}

void ActGraph::InitNodes() {
  HashMap<int64_t, const TaskProto*> actor_id2task_proto;
  for (const TaskProto& task : plan().task()) {
    actor_id2task_proto[task.task_id()] = &task;
  }
  HashMap<std::string, const ActNode*> regst_uid2producer_node;
  for (const ActEvent& act_event : *act_events_) {
    int64_t actor_id = act_event.actor_id();
    int64_t act_id = act_event.act_id();
    ActNode* producer_act_node =
        new ActNode(&act_event, actor_id2task_proto.at(actor_id));
    AddAllocatedNode(producer_act_node);
    producer_act_node->ForEachProducedRegstDescId([&](int64_t regst_desc_id) {
      const auto& regst_uid = GenRegstUid(regst_desc_id, act_id);
      regst_uid2producer_node_.insert({regst_uid, producer_act_node});
    });
  }
}

void ActGraph::InitEdges() {
  ForEachNode([&](ActNode* node) {
    for (const auto& readable : node->act_event().readable_regst_infos()) {
      const auto& regst_uid =
          GenRegstUid(readable.regst_desc_id(), readable.act_id());
      const auto& producer_it = regst_uid2producer_node_.find(regst_uid);
      if (producer_it == regst_uid2producer_node_.end()) { continue; }
      Connect(producer_it->second, NewEdge(), node);
      regst_uid2consumer_nodes_[regst_uid].push_back(node);
      producer_it->second->AddConsumerNode(regst_uid, node);
    }
  });
}

void ActGraph::TopoForEachActNode(
    const std::list<ActNode*>& starts,
    const std::function<void(ActNode*)>& Handler) const {
  TODO();
}

void ActGraph::InitDepth() {
  std::list<ActNode*> sources;
  ForEachNode([&](ActNode* node) {
    if (node->in_edges().empty()) { sources.push_back(node); }
  });
  TopoForEachActNode(sources, [&](ActNode* act_node) {
    int64_t depth = -1;
    act_node->ForEachNodeOnInEdge([&](const ActNode* in_node) {
      depth = std::max(depth, in_node->depth());
    });
    ++depth;
    act_node->set_depth(depth);
    depth2nodes_[depth].push_back(act_node);
  });
}

ActGraph::ActGraph(const Plan& plan,
                   std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  InitNodes();
  InitEdges();
  InitDepth();
}

void ActGraph::ForEachDepthRangeRegstUids(
    const std::function<void(const Range& range,
                             const std::list<std::string>& regst_uids)>&
        Handler) const {
  HashMap<Range, std::list<std::string>> depth_range2regst_uids;
  for (const auto& pair : regst_uid2producer_node_) {
    const auto& consumer_nodes_it = regst_uid2consumer_nodes_.find(pair.first);
    if (consumer_nodes_it == regst_uid2consumer_nodes_.end()) { continue; }
    int64_t begin = std::numeric_limits<int64_t>::max();
    pair.second->ForEachNodeOnOutEdge(
        [&](const ActNode* node) { begin = std::min(begin, node->depth()); });
    int64_t end = 0;
    for (const ActNode* node : consumer_nodes_it->second) {
      end = std::max(end, node->depth());
    }
    depth_range2regst_uids[Range(begin, end)].push_back(pair.first);
  }
  for (const auto& pair : depth_range2regst_uids) {
    Handler(pair.first, pair.second);
  }
}

void ActGraph::ToDotFiles(const std::string& dir) const { TODO(); }
}  // namespace oneflow
