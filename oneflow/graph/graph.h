#ifndef ONEFLOW_GRAPH_GRAPH_H_
#define ONEFLOW_GRAPH_GRAPH_H_

#include <iostream>
#include <fstream>
#include <queue>
#include "graph/node.h"

namespace oneflow {

template<typename NodeType, typename EdgeType>
class Graph {
 public:
  // (Topologically)/(Reverse Topologically) ergodic all nodes
  class Iterator;
  class ConstIterator;
  class ReverseIterator;
  class ConstReverseIterator;

  OF_DISALLOW_COPY_AND_MOVE(Graph);
  Graph() = default;
  virtual ~Graph() = default;

  // begin, end
  Iterator begin() { return Iterator(source_nodes_); }
  Iterator end() { return Iterator(std::unordered_set<NodeType*> ()); }
  ConstIterator begin() const { return cbegin(); }
  ConstIterator end() const { return cend(); }
  ConstIterator cbegin() const;
  ConstIterator cend() const;

  ReverseIterator rbegin() { return ReverseIterator(sink_nodes_); }
  ReverseIterator rend() { return ReverseIterator(std::unordered_set<NodeType*> ()); }
  ConstReverseIterator rbegin() const { return crbegin(); }
  ConstReverseIterator rend() const { return crend(); }
  ConstReverseIterator crbegin() const;
  ConstReverseIterator crend() const;
  
  // Getters
  const std::vector<std::unique_ptr<NodeType>>& nodes() const { return nodes_; }
  const std::vector<std::unique_ptr<EdgeType>>& edges() const { return edges_; }
  const std::unordered_set<NodeType*>& source_nodes() const {
    return source_nodes_;
  }
  const std::unordered_set<NodeType*>& sink_nodes() const {
    return sink_nodes_;
  }

  NodeType* SoleSourceNode() const {
    CHECK_EQ(source_nodes_.size(), 1);
    return *(source_nodes_.begin());
  }
  NodeType* SoleSinkNode() const {
    CHECK_EQ(sink_nodes_.size(), 1);
    return *(sink_nodes_.begin());
  }
  NodeType* SoleNode() const {
    CHECK_EQ(nodes_.size(), 1);
    return nodes_.front().get();
  }
  
  // Setters
  NodeType* NewNode() {
    NodeType* ret = new NodeType;
    EnrollNode(ret);
    return ret;
  }
  EdgeType* NewEdge() {
    EdgeType* ret = new EdgeType;
    EnrollEdge(ret);
    return ret;
  }
  void UpdateSourceAndSink();

  // ToDot
  virtual std::string ToDotString() const;
  void ToDotFile(const std::string& dot_filepath) const;

 protected:
  // Enroll
  void EnrollNode(NodeType* new_node) {
    nodes_.emplace_back(new_node);
  }
  void EnrollNode(std::unique_ptr<NodeType>&& new_node) {
    nodes_.push_back(std::move(new_node));
  }
  void EnrollEdge(EdgeType* new_edge) {
    edges_.emplace_back(new_edge);
  }
  void EnrollEdge(std::unique_ptr<EdgeType>&& new_edge) {
    edges_.push_back(std::move(new_edge));
  }

 private:
  std::unordered_set<NodeType*> source_nodes_;
  std::unordered_set<NodeType*> sink_nodes_;
  
  // manage delete of all nodes, edges
  std::vector<std::unique_ptr<NodeType>> nodes_;
  std::vector<std::unique_ptr<EdgeType>> edges_;
};


template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::Iterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Iterator);
  Iterator() = default;
  ~Iterator() = default;
  
  Iterator(const std::unordered_set<NodeType*>& source_nodes) {
    for (NodeType* node : source_nodes) {
      bfs_queue_.push(node);
    }
  }
  
  NodeType& operator * ();
  NodeType* operator -> ();
  Iterator& operator ++ ();
  
  bool operator != (const Iterator&) const;

 private:
  std::queue<NodeType*> bfs_queue_;
  HashMap<NodeType*, int32_t> visited_cnt_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ConstIterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ConstIterator);
  ConstIterator() = default;
  ~ConstIterator() = default;
  
  ConstIterator(const Iterator& graph_iterator) {
    graph_iterator_ = graph_iterator;
  }
  
  const NodeType& operator * () { return *graph_iterator_; }
  const NodeType* operator -> () { return &(*graph_iterator_); }
  ConstIterator& operator ++ () {
    ++graph_iterator_;
    return *this;
  }
  bool operator != (const ConstIterator& rhs) const {
    return graph_iterator_ != rhs.graph_iterator_;
  }

 private:
  Iterator graph_iterator_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ReverseIterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ReverseIterator);
  ReverseIterator() = default;
  ~ReverseIterator() = default;
  
  ReverseIterator(const std::unordered_set<NodeType*>& sink_nodes) {
    for (NodeType* node : sink_nodes) {
      bfs_queue_.push(node);
    }
  }
  
  NodeType& operator * ();
  NodeType* operator -> ();
  ReverseIterator& operator ++ ();
  
  bool operator != (const ReverseIterator&) const;

 private:
  std::queue<NodeType*> bfs_queue_;
  HashMap<NodeType*, int32_t > visited_cnt_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ConstReverseIterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ConstReverseIterator);
  ConstReverseIterator() = default;
  ~ConstReverseIterator() = default;
  
  ConstReverseIterator(const ReverseIterator& graph_iterator) {
    graph_iterator_ = graph_iterator;
  }
  
  const NodeType& operator * () { return *graph_iterator_; }
  const NodeType* operator -> () { return &(*graph_iterator_); }
  ConstReverseIterator& operator ++ () {
    ++graph_iterator_;
    return *this;
  }
  bool operator != (const ConstReverseIterator& rhs) const {
    return graph_iterator_ != rhs.graph_iterator_;
  }

 private:
  ReverseIterator graph_iterator_;
};

template<typename NodeType, typename EdgeType>
std::string Graph<NodeType, EdgeType>::ToDotString() const {
  std::stringstream ss;
  ss << "digraph {" << std::endl;
  for (const auto& node : nodes_) {
    ss << "\"" << node->VisualStr() << "\"" << std::endl;
  }
  for (const auto& edge : edges_) {
    ss << "\"" << edge->src_node()->VisualStr() << "\" -> "
       << "\"" << edge->dst_node()->VisualStr() << "\""
       << "[label=\"" << edge->VisualStr() << "\"];"
       << std::endl;
  }
  ss << "}" << std::endl;
  return ss.str();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotFile(const std::string& dot_filepath) const {
  std::fstream fs(dot_filepath.c_str(), std::fstream::out);
  CHECK(fs.good()) << "failed to open " << dot_filepath;
  fs << ToDotString();
  CHECK(fs.good()) << "failed to write " << dot_filepath;
  fs.close();
  LOG(INFO) << "Done: " << dot_filepath;
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::UpdateSourceAndSink() {
  source_nodes_.clear();
  sink_nodes_.clear();
  for (const std::unique_ptr<NodeType>& node : nodes_) {
    if (node->in_edges().empty()) {
      source_nodes_.insert(node.get());
    }
    if (node->out_edges().empty()) {
      sink_nodes_.insert(node.get());
    }
  }
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::cbegin() const -> ConstIterator{
  return ConstIterator((const_cast<Graph*>(this))->begin());
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::cend() const -> ConstIterator {
  return ConstIterator((const_cast<Graph*>(this))->end());
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::crbegin() const -> ConstReverseIterator {
  return ConstReverseIterator((const_cast<Graph*>(this))->rbegin());
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::crend() const -> ConstReverseIterator {
  return ConstReverseIterator((const_cast<Graph*>(this))->rend());
}

template<typename NodeType, typename EdgeType>
NodeType& Graph<NodeType, EdgeType>::Iterator::operator * () {
  return *(bfs_queue_.front());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::Iterator::operator -> () {
  return &(*(*this));
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::Iterator::operator ++ () -> Iterator& {
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

template<typename NodeType>
bool IsNotEqual4BfsQueue(const std::queue<NodeType*>& lhs,
                         const std::queue<NodeType*>& rhs) {
  if (lhs.empty() != rhs.empty()) {
    return true;
  }
  if (lhs.empty() == false && rhs.empty() == false) {
    return lhs.front() != rhs.front();
  }
  return false;
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::Iterator::operator != (
    const Iterator& rhs) const {
  return IsNotEqual4BfsQueue(bfs_queue_, rhs.bfs_queue_);
}

template<typename NodeType, typename EdgeType>
NodeType& Graph<NodeType, EdgeType>::ReverseIterator::operator * () {
  return *(bfs_queue_.front());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::ReverseIterator::operator -> () {
  return &(*(*this));
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::ReverseIterator::operator ++ () -> ReverseIterator& {
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
bool Graph<NodeType, EdgeType>::ReverseIterator::operator != (
    const ReverseIterator& rhs) const {
  return IsNotEqual4BfsQueue(bfs_queue_, rhs.bfs_queue_);
}

} // namespace oneflow

#endif // ONEFLOW_GRAPH_GRAPH_H_
