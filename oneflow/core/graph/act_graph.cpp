#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
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

class RegstActSubGraph final : public Graph<const ActNode, const ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstActSubGraph);
  RegstActSubGraph(const std::string& regst_uid, const ActNode* producer_node,
                   const HashSet<const ActNode*>& partial_producer_outs,
                   const HashSet<const ActNode*>& partial_consumer_nodes,
                   const std::list<const ActNode*>& fake_sources_super_set);
  ~RegstActSubGraph() = default;

  void ForEachConsumerPathDuration(
      const std::function<void(int64_t, double)>& Handler) const;

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
  bool IsWithinDepthRange(const ActNode* node) const;

  std::string regst_uid_;
  const ActNode* producer_node_;
  HashSet<const ActNode*> partial_producer_outs_;
  HashSet<const ActNode*> partial_consumer_nodes_;
  HashSet<const ActNode*> fake_sources_;
  Range depth_range_;
};

bool RegstActSubGraph::IsWithinDepthRange(const ActNode* node) const {
  return node->depth() >= depth_range_.begin()
         && node->depth() <= depth_range_.end();
}

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
  int64_t min_depth = std::numeric_limits<int64_t>::max();
  for (const ActNode* node : fake_sources_super_set) {
    min_depth = std::min(min_depth, node->depth());
  }
  int64_t max_depth = std::numeric_limits<int64_t>::min();
  for (const ActNode* node : partial_consumer_nodes_) {
    max_depth = std::max(max_depth, node->depth());
  }
  depth_range_.mut_begin() = min_depth;
  depth_range_.mut_end() = max_depth;
}

std::list<const ActNode*> RegstActSubGraph::CalcSources() const {
  std::list<const ActNode*> sources;
  for (const ActNode* node : fake_sources_) { sources.push_back(node); }
  sources.push_back(producer_node_);
  return sources;
}

void RegstActSubGraph::TopoForEachActNode(
    const std::function<void(const ActNode*)>& Handler) const {
  auto ForEachIn = std::bind(&RegstActSubGraph::ForEachInNode, this,
                             std::placeholders::_1, std::placeholders::_2);
  auto ForEachOut = std::bind(&RegstActSubGraph::ForEachOutNode, this,
                              std::placeholders::_1, std::placeholders::_2);
  TopoForEachNode(CalcSources(), ForEachIn, ForEachOut, Handler);
}

void RegstActSubGraph::ForEachInNode(
    const ActNode* node,
    const std::function<void(const ActNode*)>& Handler) const {
  if (partial_producer_outs_.find(node) != partial_producer_outs_.end()) {
    Handler(producer_node_);
  }
  node->ForEachNodeOnInEdge([&](const ActNode* node) {
    if (IsWithinDepthRange(node)) { Handler(node); }
  });
}

void RegstActSubGraph::ForEachOutNode(
    const ActNode* node,
    const std::function<void(const ActNode*)>& Handler) const {
  if (node == producer_node_) {
    for (const ActNode* producer_out : partial_producer_outs_) {
      Handler(producer_out);
    }
  } else {
    node->ForEachNodeOnOutEdge([&](const ActNode* node) {
      if (IsWithinDepthRange(node)) { Handler(node); }
    });
  }
}

void RegstActSubGraph::ForEachConsumerPathDuration(
    const std::function<void(int64_t consumer_actor_id, double duration)>&
        Handler) const {
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
  for (const ActNode* node : partial_consumer_nodes_) {
    Handler(node->actor_id(), node2longest_path_duration.at(node));
  }
}

class DepthRangeActSubGraph final : public Graph<const ActNode, const ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DepthRangeActSubGraph);
  DepthRangeActSubGraph(const ActGraph* act_graph, const Range& depth_range,
                        const std::list<std::string>& regst_uids);
  ~DepthRangeActSubGraph() = default;

  void ForEachRegstActSubGraph(
      const std::function<void(const RegstActSubGraph&)>& Handler) const;
  void ToDotFiles(const std::string& dir) const;

 private:
  void InitNode2ComponentId();
  void InitComponentId2Sources();
  void ForEachActNode(const std::list<const ActNode*>& sources,
                      const std::function<void(const ActNode*)>& Handler) const;
  void TopoForEachActNode(
      const std::list<const ActNode*>& starts,
      const std::function<void(const ActNode*)>& Handler) const;
  void ForEachInNode(const ActNode* node,
                     const std::function<void(const ActNode*)>& Handler) const;
  void ForEachOutNode(const ActNode* node,
                      const std::function<void(const ActNode*)>& Handler) const;
  void ComponentToDotFiles(const std::string& dir, int64_t component_id) const;

  const ActGraph* act_graph_;
  Range depth_range_;
  std::list<std::string> regst_uids_;
  HashMap<const ActNode*, int64_t> node2component_id_;
  HashMap<int64_t, std::list<const ActNode*>> compo_id2sources_;
  HashMap<int64_t, std::list<RegstActSubGraph>> compo_id2regst_act_graph_;
};

void DepthRangeActSubGraph::ComponentToDotFiles(const std::string& dir,
                                                int64_t component_id) const {
  std::string filepath = JoinPath(dir, std::to_string(component_id) + ".dot");
  const auto& sources = compo_id2sources_.at(component_id);
  PersistentOutStream out_stream(LocalFS(), filepath);
  out_stream << "digraph {\n";
  ForEachActNode(sources, [&](const ActNode* node) {
    out_stream << node->node_id_str() << "[label=\"" << node->VisualStr()
               << "\", shape=ellipse, style=\"rounded,filled\", "
               << "colorscheme=set312, color="
               << task_type2color.at(node->task_type()) << "];\n";
    for (const ActEdge* act_edge : node->out_edges()) {
      out_stream << "\"" << act_edge->src_node()->node_id_str() << "\" -> \""
                 << act_edge->dst_node()->node_id_str() << "\";\n";
    }
  });
  out_stream << "}\n";
}

void DepthRangeActSubGraph::ToDotFiles(const std::string& dir) const {
  std::string sub_dir = JoinPath(dir, std::to_string(depth_range_.begin()) + "-"
                                          + std::to_string(depth_range_.end()));
  LocalFS()->RecursivelyCreateDir(sub_dir);
  for (const auto& pair : compo_id2sources_) {
    ComponentToDotFiles(sub_dir, pair.first);
  }
}

void DepthRangeActSubGraph::TopoForEachActNode(
    const std::list<const ActNode*>& starts,
    const std::function<void(const ActNode*)>& Handler) const {
  auto ForEachIn = std::bind(&DepthRangeActSubGraph::ForEachInNode, this,
                             std::placeholders::_1, std::placeholders::_2);
  auto ForEachOut = std::bind(&DepthRangeActSubGraph::ForEachOutNode, this,
                              std::placeholders::_1, std::placeholders::_2);
  TopoForEachNode(starts, ForEachIn, ForEachOut, Handler);
}

void DepthRangeActSubGraph::ForEachInNode(
    const ActNode* node,
    const std::function<void(const ActNode*)>& Handler) const {
  node->ForEachNodeOnInEdge([&](ActNode* in_node) {
    if (in_node->depth() >= depth_range_.begin()
        && in_node->depth() <= depth_range_.end()) {
      Handler(in_node);
    }
  });
}

void DepthRangeActSubGraph::ForEachOutNode(
    const ActNode* node,
    const std::function<void(const ActNode*)>& Handler) const {
  node->ForEachNodeOnOutEdge([&](ActNode* out_node) {
    if (out_node->depth() >= depth_range_.begin()
        && out_node->depth() <= depth_range_.end()) {
      Handler(out_node);
    }
  });
}

void DepthRangeActSubGraph::ForEachActNode(
    const std::list<const ActNode*>& sources,
    const std::function<void(const ActNode*)>& Handler) const {
  auto ForEachConnectedNode =
      [&](const ActNode* node,
          const std::function<void(const ActNode*)>& Handler) {
        ForEachInNode(node, Handler);
        ForEachOutNode(node, Handler);
      };
  BfsForEachNode(sources, ForEachConnectedNode, Handler);
}

void DepthRangeActSubGraph::InitComponentId2Sources() {
  for (const auto& pair : node2component_id_) {
    int in_nodes_cnt = 0;
    ForEachInNode(pair.first, [&](const ActNode*) { ++in_nodes_cnt; });
    if (in_nodes_cnt == 0) {
      compo_id2sources_[pair.second].push_back(pair.first);
    }
  }
}

void DepthRangeActSubGraph::InitNode2ComponentId() {
  int64_t component_id = 0;
  const auto& sources = act_graph_->Nodes4Depth(depth_range_.begin());
  ForEachActNode(sources, [&](const ActNode* node) {
    if (node2component_id_.find(node) != node2component_id_.end()) { return; }
    ForEachActNode({node}, [&](const ActNode* component_node) {
      node2component_id_.insert({component_node, component_id});
    });
    ++component_id;
  });
}

DepthRangeActSubGraph::DepthRangeActSubGraph(
    const ActGraph* act_graph, const Range& depth_range,
    const std::list<std::string>& regst_uids)
    : act_graph_(act_graph),
      depth_range_(depth_range),
      regst_uids_(regst_uids) {
  InitNode2ComponentId();
  InitComponentId2Sources();
}

void DepthRangeActSubGraph::ForEachRegstActSubGraph(
    const std::function<void(const RegstActSubGraph&)>& Handler) const {
  for (const auto& regst_uid : regst_uids_) {
    const ActNode* producer = act_graph_->ProducerNode4RegstUid(regst_uid);
    HashMap<int64_t, HashSet<const ActNode*>> compo_id2producer_outs;
    HashMap<int64_t, HashSet<const ActNode*>> compo_id2consumers;
    HashSet<int64_t> component_ids;
    ForEachOutNode(producer, [&](const ActNode* node) {
      int64_t component_id = node2component_id_.at(node);
      compo_id2producer_outs[component_id].insert(node);
      component_ids.insert(component_id);
    });
    for (const ActNode* node : act_graph_->ConsumerNodes4RegstUid(regst_uid)) {
      int64_t component_id = node2component_id_.at(node);
      compo_id2consumers[component_id].insert(node);
      component_ids.insert(component_id);
    }
    for (int64_t compo_id : component_ids) {
      if (compo_id2producer_outs.find(compo_id) != compo_id2producer_outs.end()
          && compo_id2consumers.find(compo_id) != compo_id2consumers.end()) {
        RegstActSubGraph regst_act_graph(
            regst_uid, producer, compo_id2producer_outs.at(compo_id),
            compo_id2consumers.at(compo_id), compo_id2sources_.at(compo_id));
        Handler(regst_act_graph);
      }
    }
  }
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

void ActGraph::ForEachRegstDescConsumerPathIIScale(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  std::map<std::pair<int64_t, int64_t>, uint64_t>
      regst_desc_id_consumed2used_cnt;
  std::map<int64_t, uint64_t> regst_desc_id2produced_cnt;
  uint64_t max_cnt = 0;
  for (const auto& pair : regst_uid2consumer_nodes_) {
    int64_t regst_desc_id = RegstDescId4RegstUid(pair.first);
    int64_t produced_cnt = ++regst_desc_id2produced_cnt[regst_desc_id];
    if (max_cnt < produced_cnt) { max_cnt = produced_cnt; }
    for (const ActNode* act_node : pair.second) {
      std::pair<int64_t, int64_t> consumed_regst_desc_id(regst_desc_id,
                                                         act_node->actor_id());
      int64_t used_cnt =
          ++regst_desc_id_consumed2used_cnt[consumed_regst_desc_id];
      if (max_cnt < used_cnt) { max_cnt = used_cnt; }
    }
  }
  for (const auto& pair : regst_desc_id_consumed2used_cnt) {
    uint64_t produced_cnt = regst_desc_id2produced_cnt.at(pair.first.first);
    Handler(pair.first.first, pair.first.second,
            1.0 * max_cnt / std::min(produced_cnt, pair.second));
  }
}

void ActGraph::ForEachRegstDescConsumerPathMeanDuration(
    const std::function<void(int64_t, int64_t, double)>& Handler) const {
  std::map<std::pair<int64_t, int64_t>, double> regst_desc_id_consumed2duration;
  std::map<std::pair<int64_t, int64_t>, int> regst_desc_id_consumed2cnt;
  ForEachRegstUidConsumerPathDuration([&](const std::string& regst_uid,
                                          int64_t consumer_actor_id,
                                          double duration) {
    int64_t regst_desc_id = RegstDescId4RegstUid(regst_uid);
    std::pair<int64_t, int64_t> regst_desc_id_consumed(regst_desc_id,
                                                       consumer_actor_id);
    regst_desc_id_consumed2duration[regst_desc_id_consumed] += duration;
    ++regst_desc_id_consumed2cnt[regst_desc_id_consumed];
  });
  for (const auto& pair : regst_desc_id_consumed2duration) {
    Handler(pair.first.first, pair.first.second,
            pair.second / regst_desc_id_consumed2cnt.at(pair.first));
  }
}

void ActGraph::ForEachRegstActSubGraph(
    const std::function<void(const RegstActSubGraph&)>& Handler) const {
  ForEachDepthRangeSubActGraph([&](const DepthRangeActSubGraph& sub_graph) {
    sub_graph.ForEachRegstActSubGraph(Handler);
  });
}

void ActGraph::ForEachDepthRangeSubActGraph(
    const std::function<void(const DepthRangeActSubGraph&)>& Handler) const {
  ForEachDepthRangeRegstUids(
      [&](const Range& range, const std::list<std::string>& regst_uids) {
        DepthRangeActSubGraph depth_range_subgrpah(this, range, regst_uids);
        Handler(depth_range_subgrpah);
      });
}

void ActGraph::ForEachRegstUidConsumerPathDuration(
    const std::function<void(const std::string&, int64_t, double)>& Handler)
    const {
  ForEachRegstActSubGraph([&](const RegstActSubGraph& regst_csm_graph) {
    const std::string regst_uid = regst_csm_graph.regst_uid();
    regst_csm_graph.ForEachConsumerPathDuration(
        [&](int64_t consumer_actor_id, double duration) {
          Handler(regst_uid, consumer_actor_id, duration);
        });
  });
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
  TopoForEachNode(starts, &ActNode::ForEachNodeOnInEdge,
                  &ActNode::ForEachNodeOnOutEdge, Handler);
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

void ActGraph::InitTaskId2TaskProto() {
  for (const auto& task_proto : plan_->task()) {
    task_id2task_proto_.emplace(task_proto.task_id(), &task_proto);
  }
}

ActGraph::ActGraph(const Plan& plan,
                   std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  InitNodes();
  InitEdges();
  InitDepth();
  InitTaskId2TaskProto();
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

void ActGraph::ToDotFiles(const std::string& dir) const {
  ForEachDepthRangeSubActGraph([&](const DepthRangeActSubGraph& sub_graph) {
    sub_graph.ToDotFiles(dir);
  });
}

}  // namespace oneflow
