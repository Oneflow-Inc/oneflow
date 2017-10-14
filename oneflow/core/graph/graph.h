#ifndef ONEFLOW_CORE_GRAPH_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_GRAPH_H_

#include "gflags/gflags.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/graph/node.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

template<typename NodeType, typename EdgeType>
class Graph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Graph);
  Graph() = default;
  virtual ~Graph() = default;

  // For Each Node
  void ForEachNode(std::function<void(NodeType*)>);
  void TopoForEachNode(std::function<void(NodeType*)>);
  void ReverseTopoForEachNode(std::function<void(NodeType*)>);
  void ConstForEachNode(std::function<void(const NodeType*)>) const;
  void ConstTopoForEachNode(std::function<void(const NodeType*)>) const;
  void ConstReverseTopoForEachNode(std::function<void(const NodeType*)>) const;

  // For Each Edge
  void ForEachEdge(std::function<void(EdgeType*)>);
  void ConstForEachEdge(std::function<void(const EdgeType*)>) const;

  // Getters
  const std::unordered_set<NodeType*>& source_nodes() const;
  const std::unordered_set<NodeType*>& sink_nodes() const;
  NodeType* SoleSourceNode() const;
  NodeType* SoleSinkNode() const;
  NodeType* SoleNode() const;
  size_t node_num() const { return nodes_.size(); }
  size_t edge_num() const { return edges_.size(); }
  virtual const char* TypeName() const { return "Not Defined"; }

  // Setters
  NodeType* NewNode();
  EdgeType* NewEdge();
  void EnrollNode(NodeType*);
  void EnrollNode(std::unique_ptr<NodeType>&&);
  void EnrollEdge(EdgeType*);
  void EnrollEdge(std::unique_ptr<EdgeType>&&);
  void UpdateSourceAndSink();

  // ToDot
  template<typename StreamT>
  void ToDotWithStream(StreamT& out_stream) const;
  void ToDotWithFilePath(const std::string& file_path) const;
  void ToDotWithAutoFilePath() const;

 private:
  class TopoIterator;
  class ReverseTopoIterator;
  TopoIterator begin() { return source_nodes_; }
  TopoIterator end() { return std::unordered_set<NodeType*>(); }
  ReverseTopoIterator rbegin() { return sink_nodes_; }
  ReverseTopoIterator rend() { return std::unordered_set<NodeType*>(); }

  //
  std::unordered_set<NodeType*> source_nodes_;
  std::unordered_set<NodeType*> sink_nodes_;
  std::vector<std::unique_ptr<NodeType>> nodes_;
  std::vector<std::unique_ptr<EdgeType>> edges_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::TopoIterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(TopoIterator);
  TopoIterator() = default;
  ~TopoIterator() = default;

  TopoIterator(const std::unordered_set<NodeType*>& source_nodes) {
    for (NodeType* node : source_nodes) { bfs_queue_.push(node); }
  }

  NodeType& operator*() { return *(bfs_queue_.front()); }
  NodeType* operator->() { return &(*(*this)); }
  TopoIterator& operator++();

  bool operator!=(const TopoIterator&) const;

 private:
  std::queue<NodeType*> bfs_queue_;
  HashMap<NodeType*, int32_t> visited_cnt_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ReverseTopoIterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ReverseTopoIterator);
  ReverseTopoIterator() = default;
  ~ReverseTopoIterator() = default;

  ReverseTopoIterator(const std::unordered_set<NodeType*>& sink_nodes) {
    for (NodeType* node : sink_nodes) { bfs_queue_.push(node); }
  }

  NodeType& operator*() { return *(bfs_queue_.front()); }
  NodeType* operator->() { return &(*(*this)); }

  ReverseTopoIterator& operator++();

  bool operator!=(const ReverseTopoIterator&) const;

 private:
  std::queue<NodeType*> bfs_queue_;
  HashMap<NodeType*, int32_t> visited_cnt_;
};

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachNode(
    std::function<void(NodeType*)> func) {
  for (auto& x : nodes_) { func(x.get()); }
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::TopoForEachNode(
    std::function<void(NodeType*)> func) {
  for (TopoIterator it = begin(); it != end(); ++it) { func(&(*it)); }
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ReverseTopoForEachNode(
    std::function<void(NodeType*)> func) {
  for (ReverseTopoIterator it = rbegin(); it != rend(); ++it) { func(&(*it)); }
}

#define OF_DEFINE_CONST_FOR_EACH_NODE(FuncName)                    \
  template<typename NodeType, typename EdgeType>                   \
  void Graph<NodeType, EdgeType>::Const##FuncName(                 \
      std::function<void(const NodeType*)> func) const {           \
    auto cast_this = const_cast<Graph<NodeType, EdgeType>*>(this); \
    cast_this->FuncName(std::bind(func, std::placeholders::_1));   \
  }

OF_DEFINE_CONST_FOR_EACH_NODE(ForEachNode);
OF_DEFINE_CONST_FOR_EACH_NODE(TopoForEachNode);
OF_DEFINE_CONST_FOR_EACH_NODE(ReverseTopoForEachNode);

#undef OF_DEFINE_CONST_FOR_EACH_NODE

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachEdge(
    std::function<void(EdgeType*)> func) {
  for (auto& x : edges_) { func(x.get()); }
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ConstForEachEdge(
    std::function<void(const EdgeType*)> func) const {
  auto cast_this = const_cast<Graph<NodeType, EdgeType>*>(this);
  cast_this->ForEachEdge(std::bind(func, std::placeholders::_1));
}

template<typename NodeType, typename EdgeType>
const std::unordered_set<NodeType*>& Graph<NodeType, EdgeType>::source_nodes()
    const {
  return source_nodes_;
}

template<typename NodeType, typename EdgeType>
const std::unordered_set<NodeType*>& Graph<NodeType, EdgeType>::sink_nodes()
    const {
  return sink_nodes_;
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleSourceNode() const {
  CHECK_EQ(source_nodes_.size(), 1);
  return *(source_nodes_.begin());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleSinkNode() const {
  CHECK_EQ(sink_nodes_.size(), 1);
  return *(sink_nodes_.begin());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleNode() const {
  CHECK_EQ(nodes_.size(), 1);
  return nodes_.front().get();
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::NewNode() {
  NodeType* ret = new NodeType;
  EnrollNode(ret);
  return ret;
}

template<typename NodeType, typename EdgeType>
EdgeType* Graph<NodeType, EdgeType>::NewEdge() {
  EdgeType* ret = new EdgeType;
  EnrollEdge(ret);
  return ret;
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::EnrollNode(NodeType* node) {
  nodes_.emplace_back(node);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::EnrollNode(std::unique_ptr<NodeType>&& node) {
  nodes_.push_back(std::move(node));
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::EnrollEdge(EdgeType* edge) {
  edges_.emplace_back(edge);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::EnrollEdge(std::unique_ptr<EdgeType>&& edge) {
  edges_.push_back(std::move(edge));
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::UpdateSourceAndSink() {
  source_nodes_.clear();
  sink_nodes_.clear();
  for (const std::unique_ptr<NodeType>& node : nodes_) {
    if (node->in_edges().empty()) { source_nodes_.insert(node.get()); }
    if (node->out_edges().empty()) { sink_nodes_.insert(node.get()); }
  }
}

template<typename NodeType, typename EdgeType>
template<typename StreamT>
void Graph<NodeType, EdgeType>::ToDotWithStream(StreamT& out_stream) const {
  out_stream << "digraph {\n";
  this->ConstForEachNode([&](const NodeType* node) {
    out_stream << "\"" << node->VisualStr() << "\"\n";
  });
  this->ConstForEachEdge([&](const EdgeType* edge) {
    out_stream << "\"" << edge->src_node()->VisualStr() << "\" -> "
               << "\"" << edge->dst_node()->VisualStr() << "\""
               << "[label=\"" << edge->VisualStr() << "\"];\n";
  });
  out_stream << "}\n";
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotWithFilePath(
    const std::string& file_path) const {
  std::string dir_name = Dirname(file_path);
  if (!LocalFS()->IsDirectory(dir_name)) {
    LocalFS()->RecursivelyCreateDir(dir_name);
  }
  PersistentOutStream out_stream(LocalFS(), file_path);
  ToDotWithStream(out_stream);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotWithAutoFilePath() const {
  std::string file_path =
      LogDir() + "/dot/" + TypeName() + "/" + NewUniqueId() + ".dot";
  ToDotWithFilePath(file_path);
}

template<typename NodeType>
bool IsNotEqual4BfsQueue(const std::queue<NodeType*>& lhs,
                         const std::queue<NodeType*>& rhs) {
  if (lhs.empty() != rhs.empty()) { return true; }
  if (lhs.empty() == false && rhs.empty() == false) {
    return lhs.front() != rhs.front();
  }
  return false;
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::TopoIterator::operator++() -> TopoIterator& {
  NodeType* cur_node = bfs_queue_.front();
  bfs_queue_.pop();
  for (EdgeType* out_edge : cur_node->out_edges()) {
    NodeType* dst_node = out_edge->dst_node();
    visited_cnt_[dst_node] += 1;
    if (visited_cnt_.at(dst_node) == dst_node->in_edges().size()) {
      bfs_queue_.push(dst_node);
    }
  }
  return *this;
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::TopoIterator::operator!=(
    const TopoIterator& rhs) const {
  return IsNotEqual4BfsQueue(bfs_queue_, rhs.bfs_queue_);
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::ReverseTopoIterator::operator++()
    -> ReverseTopoIterator& {
  NodeType* cur_node = bfs_queue_.front();
  bfs_queue_.pop();
  for (EdgeType* in_edge : cur_node->in_edges()) {
    NodeType* src_node = in_edge->src_node();
    visited_cnt_[src_node] += 1;
    if (visited_cnt_.at(src_node) == src_node->out_edges().size()) {
      bfs_queue_.push(src_node);
    }
  }
  return *this;
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::ReverseTopoIterator::operator!=(
    const ReverseTopoIterator& rhs) const {
  return IsNotEqual4BfsQueue(bfs_queue_, rhs.bfs_queue_);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_GRAPH_H_
