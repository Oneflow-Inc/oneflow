#ifndef ONEFLOW_CORE_GRAPH_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_GRAPH_H_

#include <stack>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/graph/node.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

template<typename NodeType, typename EdgeType>
class Graph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Graph);
  Graph() = default;
  virtual ~Graph() = default;

  // For Each
  void ForEachNode(std::function<void(NodeType*)> NodeHandler) const;
  void TopoForEachNode(std::function<void(NodeType*)> NodeHandler) const;
  void ReverseTopoForEachNode(std::function<void(NodeType*)> NodeHandler) const;
  void ForEachEdge(std::function<void(EdgeType*)> EdgeHandler) const;

  void BfsForEachNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
      const std::function<void(NodeType*)>& Handler) const;

  void DfsForEachNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
      const std::function<void(NodeType*)>& Handler) const;

  void TopoForEachNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<void(NodeType*)>& Handler) const;

  void DfsTopoForEachNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<void(NodeType*)>& Handler) const;

  void DfsTopoForEachNodeSortByDistanceToSink(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<void(NodeType*)>& Handler) const;

  std::function<bool(const NodeType* src, const NodeType* dst)> MakePredicatorIsReachable() const;

  std::function<bool(const NodeType* src, const NodeType* dst)> MakePredicatorIsReachable(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
      const;

  void ForEachConnectedComponent(
      const std::function<void(const HashSet<NodeType*>&)>& Handler) const;

  void ForEachConnectedComponent(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachConnected,
      const std::function<void(const HashSet<NodeType*>&)>& Handler) const;

  NodeType* FindFirstBackEdgeDstNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext)
      const;

  NodeType* FindFirstBackEdgeDstNode() const;

  // Getters
  std::list<NodeType*> source_nodes() const;
  std::list<NodeType*> sink_nodes() const;
  NodeType* SoleSourceNode() const;
  NodeType* SoleSinkNode() const;
  NodeType* SoleNode() const;
  size_t node_num() const { return nodes_.size(); }
  size_t edge_num() const { return edges_.size(); }
  virtual const char* TypeName() const { return ""; }

  // Setters
  template<typename DerivedNodeType = NodeType>
  DerivedNodeType* NewNode();
  template<class... Args>
  EdgeType* NewEdge(Args&&... args);
  void AddAllocatedNode(NodeType*);
  void AddAllocatedEdge(EdgeType*);
  void DeleteNode(NodeType*);

  // ToDot
  template<typename StreamT>
  void ToDotWithStream(StreamT& out_stream);
  void ToDotWithFilePath(const std::string& file_path);
  void ToDotWithAutoFilePath();

 private:
  void ForEachConnectedComponent(
      const std::function<void(const std::function<void(NodeType*)>&)>& ForEachPotentialStart,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachConnected,
      const std::function<void(const HashSet<NodeType*>&)>& Handler) const;

  NodeType* FindFirstBackEdgeDstNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
      size_t* node_cnt) const;

  std::vector<std::unique_ptr<NodeType>> nodes_;
  std::vector<std::unique_ptr<EdgeType>> edges_;
};

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachNode(std::function<void(NodeType*)> NodeHandler) const {
  for (auto& x : nodes_) { NodeHandler(x.get()); }
}

template<typename NodeType, typename EdgeType>
std::list<NodeType*> Graph<NodeType, EdgeType>::source_nodes() const {
  std::list<NodeType*> ret;
  ForEachNode([&](NodeType* node) {
    if (node->in_edges().empty()) { ret.push_back(node); }
  });
  return ret;
}

template<typename NodeType, typename EdgeType>
std::list<NodeType*> Graph<NodeType, EdgeType>::sink_nodes() const {
  std::list<NodeType*> ret;
  ForEachNode([&](NodeType* node) {
    if (node->out_edges().empty()) { ret.push_back(node); }
  });
  return ret;
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleSourceNode() const {
  std::list<NodeType*> source_nodes_list = source_nodes();
  CHECK_EQ(source_nodes_list.size(), 1);
  return source_nodes_list.front();
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleSinkNode() const {
  std::list<NodeType*> sink_nodes_list = sink_nodes();
  CHECK_EQ(sink_nodes_list.size(), 1);
  return sink_nodes_list.front();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::TopoForEachNode(std::function<void(NodeType*)> NodeHandler) const {
  TopoForEachNode(source_nodes(), &NodeType::ForEachNodeOnInEdge, &NodeType::ForEachNodeOnOutEdge,
                  NodeHandler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ReverseTopoForEachNode(
    std::function<void(NodeType*)> NodeHandler) const {
  TopoForEachNode(sink_nodes(), &NodeType::ForEachNodeOnOutEdge, &NodeType::ForEachNodeOnInEdge,
                  NodeHandler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachEdge(std::function<void(EdgeType*)> EdgeHandler) const {
  for (auto& x : edges_) {
    if (x->src_node() == nullptr && x->dst_node() == nullptr) { continue; }
    EdgeHandler(x.get());
  }
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleNode() const {
  CHECK_EQ(nodes_.size(), 1);
  return nodes_.front().get();
}

template<typename NodeType, typename EdgeType>
template<typename DerivedNodeType>
DerivedNodeType* Graph<NodeType, EdgeType>::NewNode() {
  DerivedNodeType* ret = new DerivedNodeType;
  AddAllocatedNode(ret);
  return ret;
}

template<typename NodeType, typename EdgeType>
template<class... Args>
EdgeType* Graph<NodeType, EdgeType>::NewEdge(Args&&... args) {
  EdgeType* ret = new EdgeType(std::forward<Args>(args)...);
  AddAllocatedEdge(ret);
  return ret;
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::AddAllocatedNode(NodeType* node) {
  nodes_.emplace_back(node);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::AddAllocatedEdge(EdgeType* edge) {
  edges_.emplace_back(edge);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::DeleteNode(NodeType* node) {
  Erase<std::vector<std::unique_ptr<NodeType>>>(
      nodes_, [node](const std::unique_ptr<NodeType>& node_ptr) { return node_ptr.get() == node; });
}

template<typename NodeType, typename EdgeType>
template<typename StreamT>
void Graph<NodeType, EdgeType>::ToDotWithStream(StreamT& out_stream) {
  out_stream << "digraph {\n";
  this->ForEachNode([&](NodeType* node) {
    out_stream << "\"" << node->node_id_str() << "\" [label=\"" << node->VisualStr() << "\"]\n";
  });
  this->ForEachEdge([&](const EdgeType* edge) {
    out_stream << "\"" << edge->src_node()->node_id_str() << "\" -> "
               << "\"" << edge->dst_node()->node_id_str() << "\""
               << "[label=\"" << edge->VisualStr() << "\"];\n";
  });
  out_stream << "}\n";
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotWithFilePath(const std::string& file_path) {
  auto log_stream = TeePersistentLogStream::Create(file_path);
  ToDotWithStream(log_stream);
  log_stream->Flush();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotWithAutoFilePath() {
  std::string file_path = JoinPath("dot", TypeName(), NewUniqueId() + ".dot");
  ToDotWithFilePath(file_path);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::BfsForEachNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
    const std::function<void(NodeType*)>& Handler) const {
  HashSet<NodeType*> queued_nodes;
  std::queue<NodeType*> queue;
  for (NodeType* start : starts) {
    if (queued_nodes.find(start) == queued_nodes.end()) {
      queue.push(start);
      queued_nodes.insert(start);
    }
  }
  while (!queue.empty()) {
    NodeType* cur_node = queue.front();
    queue.pop();
    Handler(cur_node);
    ForEachNext(cur_node, [&](NodeType* next) {
      if (queued_nodes.find(next) == queued_nodes.end()) {
        queue.push(next);
        queued_nodes.insert(next);
      }
    });
  }
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::DfsForEachNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
    const std::function<void(NodeType*)>& Handler) const {
  HashSet<NodeType*> visited_nodes;
  std::stack<NodeType*> stack;
  for (NodeType* start : starts) { stack.push(start); }
  while (!stack.empty()) {
    NodeType* cur_node = stack.top();
    stack.pop();
    if (visited_nodes.find(cur_node) == visited_nodes.end()) {
      Handler(cur_node);
      visited_nodes.insert(cur_node);
      ForEachNext(cur_node, [&](NodeType* next) {
        if (visited_nodes.find(next) == visited_nodes.end()) { stack.push(next); }
      });
    }
  }
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::FindFirstBackEdgeDstNode() const {
  if (nodes_.empty()) { return nullptr; }
  const auto& starts = source_nodes();
  if (starts.empty()) { return nodes_.at(0).get(); }
  size_t node_cnt = 0;
  auto ForEachNext = &NodeType::ForEachNodeOnOutEdge;
  NodeType* ret = FindFirstBackEdgeDstNode(starts, ForEachNext, &node_cnt);
  if (ret == nullptr && node_cnt != nodes_.size()) {
    HashSet<NodeType*> visited_nodes;
    BfsForEachNode(starts, ForEachNext, [&](NodeType* node) { visited_nodes.emplace(node); });
    for (const auto& node : nodes_) {
      if (visited_nodes.find(node.get()) == visited_nodes.end()) { return node.get(); }
    }
  }
  return ret;
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::FindFirstBackEdgeDstNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext)
    const {
  size_t node_cnt = 0;
  return FindFirstBackEdgeDstNode(starts, ForEachNext, &node_cnt);
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::FindFirstBackEdgeDstNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
    size_t* node_cnt) const {
  NodeType* back_edge_dst_node = nullptr;
  HashSet<NodeType*> visited_nodes;
  *node_cnt = 0;
  DfsForEachNode(starts, ForEachNext, [&](NodeType* node) {
    ++*node_cnt;
    if (back_edge_dst_node != nullptr) { return; }
    visited_nodes.emplace(node);
    ForEachNext(node, [&](NodeType* next_node) {
      if (back_edge_dst_node != nullptr) { return; }
      if (visited_nodes.find(next_node) != visited_nodes.end()) { back_edge_dst_node = next_node; }
    });
  });
  return back_edge_dst_node;
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::TopoForEachNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<void(NodeType*)>& Handler) const {
  HashMap<NodeType*, bool> has_queued;
  std::queue<NodeType*> queue;
  for (NodeType* start : starts) {
    queue.push(start);
    has_queued[start] = true;
    ForEachInNode(start, [&](NodeType*) { LOG(FATAL) << "not a source"; });
  }
  while (!queue.empty()) {
    NodeType* cur_node = queue.front();
    queue.pop();
    Handler(cur_node);
    ForEachOutNode(cur_node, [&](NodeType* out) {
      bool is_ready = true;
      ForEachInNode(out, [&](NodeType* in) {
        if (is_ready && !has_queued[in]) { is_ready = false; }
      });
      if (is_ready && !has_queued[out]) {
        queue.push(out);
        has_queued[out] = true;
      }
    });
  }
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::DfsTopoForEachNodeSortByDistanceToSink(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<void(NodeType*)>& Handler) const {
  HashMap<NodeType*, int64_t> node2distance_to_sink;
  {
    std::list<NodeType*> nodes;
    TopoForEachNode(starts, ForEachInNode, ForEachOutNode,
                    [&](NodeType* node) { nodes.push_back(node); });
    std::list<NodeType*> sinks;
    for (NodeType* node : nodes) {
      bool is_sink = true;
      ForEachOutNode(node, [&](NodeType* out_node) { is_sink = false; });
      if (is_sink) { sinks.push_back(node); }
    }
    TopoForEachNode(sinks, ForEachOutNode, ForEachInNode, [&](NodeType* node) {
      int64_t distance_to_sink = -1;
      ForEachOutNode(node, [&](NodeType* out_node) {
        distance_to_sink = std::max(distance_to_sink, node2distance_to_sink[out_node]);
      });
      node2distance_to_sink[node] = distance_to_sink + 1;
    });
  }
  auto ForEachOutNodeSortedByDistanceToSink = [&](NodeType* node,
                                                  const std::function<void(NodeType*)>& Handler) {
    std::vector<NodeType*> out_nodes;
    ForEachOutNode(node, [&](NodeType* out_node) { out_nodes.push_back(out_node); });
    std::sort(out_nodes.begin(), out_nodes.end(), [&](NodeType* lhs, NodeType* rhs) {
      // DfsTopoForEachNode use stack, so sort desc
      return node2distance_to_sink.at(lhs) > node2distance_to_sink.at(rhs);
    });
    for (NodeType* out_node : out_nodes) { Handler(out_node); }
  };
  DfsTopoForEachNode(starts, ForEachInNode, ForEachOutNodeSortedByDistanceToSink, Handler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::DfsTopoForEachNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<void(NodeType*)>& Handler) const {
  HashMap<NodeType*, bool> be_visited;
  std::stack<NodeType*> stack;
  for (NodeType* start : starts) {
    stack.push(start);
    ForEachInNode(start, [&](NodeType*) { LOG(FATAL) << "not a source"; });
  }
  while (!stack.empty()) {
    NodeType* cur_node = stack.top();
    stack.pop();
    Handler(cur_node);
    be_visited[cur_node] = true;
    ForEachOutNode(cur_node, [&](NodeType* out) {
      bool is_ready = true;
      ForEachInNode(out, [&](NodeType* in) {
        if (is_ready && !be_visited[in]) { is_ready = false; }
      });
      if (is_ready && !be_visited[out]) { stack.push(out); }
    });
  }
}

template<typename NodeType, typename EdgeType>
std::function<bool(const NodeType* src, const NodeType* dst)>
Graph<NodeType, EdgeType>::MakePredicatorIsReachable() const {
  return MakePredicatorIsReachable(source_nodes(), &NodeType::ForEachNodeOnInEdge,
                                   &NodeType::ForEachNodeOnOutEdge);
}

template<typename NodeType, typename EdgeType>
std::function<bool(const NodeType* src, const NodeType* dst)>
Graph<NodeType, EdgeType>::MakePredicatorIsReachable(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
    const {
  auto node2ancestor = std::make_shared<HashMap<const NodeType*, HashSet<const NodeType*>>>();
  TopoForEachNode(starts, ForEachInNode, ForEachOutNode, [&](NodeType* node) {
    ForEachInNode(node, [&](NodeType* in_node) {
      (*node2ancestor)[node].insert(in_node);
      (*node2ancestor)[node].insert((*node2ancestor)[in_node].begin(),
                                    (*node2ancestor)[in_node].end());
    });
  });
  return [node2ancestor](const NodeType* src, const NodeType* dst) -> bool {
    return node2ancestor->at(dst).find(src) != node2ancestor->at(dst).end();
  };
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachConnectedComponent(
    const std::function<void(const HashSet<NodeType*>&)>& Handler) const {
  ForEachConnectedComponent(
      [&](const std::function<void(NodeType*)>& Handler) { ForEachNode(Handler); },
      &NodeType::ForEachNodeOnInOutEdge, Handler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachConnectedComponent(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachConnected,
    const std::function<void(const HashSet<NodeType*>&)>& Handler) const {
  auto ForEachPotentialStart = [&](const std::function<void(NodeType*)>& Handler) {
    BfsForEachNode(starts, ForEachConnected, Handler);
  };
  ForEachConnectedComponent(ForEachPotentialStart, ForEachConnected, Handler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachConnectedComponent(
    const std::function<void(const std::function<void(NodeType*)>&)>& ForEachPotentialStart,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachConnected,
    const std::function<void(const HashSet<NodeType*>&)>& Handler) const {
  HashMap<NodeType*, int32_t> node2component_id;
  int32_t cur_component_id = 0;
  ForEachPotentialStart([&](NodeType* start) {
    if (node2component_id.find(start) != node2component_id.end()) { return; }
    ++cur_component_id;
    BfsForEachNode({start}, ForEachConnected, [&](NodeType* node) {
      CHECK(node2component_id.emplace(node, cur_component_id).second);
    });
  });
  HashMap<int32_t, HashSet<NodeType*>> component_id2nodes;
  for (const auto& pair : node2component_id) { component_id2nodes[pair.second].insert(pair.first); }
  for (const auto& pair : component_id2nodes) { Handler(pair.second); }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_GRAPH_H_
