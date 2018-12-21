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
  EdgeType* NewEdge();
  void AddAllocatedNode(NodeType*);
  void AddAllocatedEdge(EdgeType*);
  void DeleteNode(NodeType*);

  // ToDot
  template<typename StreamT>
  void ToDotWithStream(StreamT& out_stream);
  void ToDotWithFilePath(const std::string& file_path);
  void ToDotWithAutoFilePath();

 private:
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
EdgeType* Graph<NodeType, EdgeType>::NewEdge() {
  EdgeType* ret = new EdgeType;
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
  HashMap<NodeType*, bool> has_queued;
  std::queue<NodeType*> queue;
  for (NodeType* start : starts) {
    queue.push(start);
    has_queued[start] = true;
  }
  while (!queue.empty()) {
    NodeType* cur_node = queue.front();
    queue.pop();
    Handler(cur_node);
    ForEachNext(cur_node, [&](NodeType* next) {
      if (!has_queued[next]) {
        queue.push(next);
        has_queued[next] = true;
      }
    });
  }
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
  std::list<NodeType*> nodes;
  TopoForEachNode(starts, ForEachInNode, ForEachOutNode,
                  [&](NodeType* node) { nodes.push_back(node); });
  std::list<NodeType*> sinks;
  for (NodeType* node : nodes) {
    bool is_sink = true;
    ForEachOutNode(node, [&](NodeType* out_node) { is_sink = false; });
    if (is_sink) { sinks.push_back(node); }
  }
  HashMap<NodeType*, int64_t> node2distance_to_sink;
  TopoForEachNode(sinks, ForEachOutNode, ForEachInNode, [&](NodeType* node) {
    int64_t distince_to_sink = -1;
    ForEachOutNode(node, [&](NodeType* out_node) {
      distince_to_sink = std::max(distince_to_sink, node2distance_to_sink[out_node]);
    });
    node2distance_to_sink[node] = distince_to_sink + 1;
  });
  auto ForEachOutNodeSortedByDistanceToSink = [&](NodeType* node,
                                                  const std::function<void(NodeType*)>& Handler) {
    std::vector<NodeType*> out_nodes;
    ForEachOutNode(node, [&](NodeType* out_node) { out_nodes.push_back(out_node); });
    std::sort(out_nodes.begin(), out_nodes.end(), [&](NodeType* lhs, NodeType* rhs) {
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

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_GRAPH_H_
