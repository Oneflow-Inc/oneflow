#ifndef ONEFLOW_GRAPH_GRAPH_H_
#define ONEFLOW_GRAPH_GRAPH_H_

#include <iostream>
#include <queue>
#include "graph/node.h"

namespace oneflow {

template<typename NodeType, typename EdgeType>
class Graph {
 public:
  // Topologically ergodic all nodes except source_node_,sink_node_
  class Iterator;
  class ConstIterator;
  // Reverse Topologically ergodic all nodes except source_node_,sink_node_
  class ReverseIterator;
  class ConstReverseIterator;

  OF_DISALLOW_COPY_AND_MOVE(Graph);
  Graph() = default;
  virtual ~Graph() = default;

  const NodeType& source_node() const { return source_node_; }
  const NodeType& sink_node() const { return sink_node_; }

  Iterator begin();
  Iterator end();
  ConstIterator begin() const { return cbegin(); }
  ConstIterator end() const { return cend(); }
  ConstIterator cbegin() const;
  ConstIterator cend() const;

  ReverseIterator rbegin();
  ReverseIterator rend();
  ConstReverseIterator rbegin() const { return crbegin(); }
  ConstReverseIterator rend() const { return crend(); }
  ConstReverseIterator crbegin() const;
  ConstReverseIterator crend() const;
  
  const std::vector<std::unique_ptr<NodeType>>& nodes() const {
    return nodes_;
  }
  const std::vector<std::unique_ptr<EdgeType>>& edges() const {
    return edges_;
  }
  
  bool IsFirstNode(const NodeType* node) const {
    return node->SoleInEdge()->src_node() == &source_node_;
  }
  bool IsLastNode(const NodeType* node) const {
    return node->SoleOutEdge()->dst_node() == &sink_node_;
  }
  
  void UpdateSourceAndSink();

 protected:
  // Register
  void RegisterNode(NodeType* new_node) {
    nodes_.emplace_back(new_node);
  }
  void RegisterNode(std::unique_ptr<NodeType>&& new_node) {
    nodes_.push_back(std::move(new_node));
  }
  void RegisterEdge(EdgeType* new_edge) {
    edges_.emplace_back(new_edge);
  }
  void RegisterEdge(std::unique_ptr<EdgeType>&& new_node) {
    edges_.push_back(std::move(new_node));
  }
  // New
  NodeType* NewFinalNode() {
    // In c++14, we can use std::is_final to check
    NodeType* ret = new NodeType;
    RegisterNode(ret);
    return ret;
  }
  EdgeType* NewFinalEdge() {
    // In c++14, we can use std::is_final to check
    EdgeType* ret = new EdgeType;
    RegisterEdge(ret);
    return ret;
  }

 private:
  NodeType source_node_;
  NodeType sink_node_;
  std::vector<std::unique_ptr<EdgeType>> source_edges_;
  std::vector<std::unique_ptr<EdgeType>> sink_edges_;
  
  // manage nodes,edges that are not related to source,sink
  std::vector<std::unique_ptr<NodeType>> nodes_;
  std::vector<std::unique_ptr<EdgeType>> edges_;
};


template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::Iterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Iterator);
  Iterator() = default;
  ~Iterator() = default;
  
  void Init(NodeType* source_node) {
    bfs_queue_ = std::queue<NodeType*> ();
    bfs_queue_.push(source_node);
  }
  
  NodeType& operator * ();
  NodeType* operator -> ();
  Iterator& operator ++ ();
  
  bool operator != (const Iterator&) const;

 private:
  std::queue<NodeType*> bfs_queue_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ConstIterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ConstIterator);
  ConstIterator() = default;
  ~ConstIterator() = default;
  
  void Init(const Iterator& graph_iterator) {
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
  
  void Init(NodeType* sink_node) {
    bfs_queue_ = std::queue<NodeType*> ();
    bfs_queue_.push(sink_node);
  }
  
  NodeType& operator * ();
  NodeType* operator -> ();
  ReverseIterator& operator ++ ();
  
  bool operator != (const ReverseIterator&) const;

 private:
  std::queue<NodeType*> bfs_queue_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ConstReverseIterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ConstReverseIterator);
  ConstReverseIterator() = default;
  ~ConstReverseIterator() = default;
  
  void Init(const ReverseIterator& graph_iterator) {
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
void Graph<NodeType, EdgeType>::UpdateSourceAndSink() {
  source_node_.DisconnectAllEdges();
  sink_node_.DisconnectAllEdges();
  source_edges_.clear();
  sink_edges_.clear();
  for (const std::unique_ptr<NodeType>& node : nodes_) {
    if (node->in_edges().empty()) {
      EdgeType* source_edge = new EdgeType;
      source_edges_.emplace_back(source_edge);
      Connect(&source_node_, source_edge, node.get());
    }
    if (node->out_edges().empty()) {
      EdgeType* sink_edge = new EdgeType;
      sink_edges_.emplace_back(sink_edge);
      Connect(node.get(), sink_edge, &sink_node_);
    }
  }
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::begin() -> Iterator {
  Iterator ret;
  ret.Init(&source_node_);
  return ++ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::end() -> Iterator {
  Iterator ret;
  ret.Init(&sink_node_);
  return ret;
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::cbegin() const -> ConstIterator{
  ConstIterator ret;
  ret.Init((const_cast<Graph*>(this))->begin());
  return ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::cend() const -> ConstIterator {
  ConstIterator ret;
  ret.Init((const_cast<Graph*>(this))->end());
  return ret;
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::rbegin() -> ReverseIterator {
  ReverseIterator ret;
  ret.Init(&sink_node_);
  return ++ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::rend() -> ReverseIterator {
  ReverseIterator ret;
  ret.Init(&source_node_);
  return ret;
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::crbegin() const -> ConstReverseIterator {
  ConstReverseIterator ret;
  ret.Init((const_cast<Graph*>(this))->rbegin());
  return ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::crend() const -> ConstReverseIterator {
  ConstReverseIterator ret;
  ret.Init((const_cast<Graph*>(this))->rend());
  return ret;
}

template<typename NodeType, typename EdgeType>
NodeType& Graph<NodeType, EdgeType>::Iterator::operator * () {
  CHECK_EQ(bfs_queue_.empty(), false);
  return *(bfs_queue_.front());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::Iterator::operator -> () {
  return &(*(*this));
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::Iterator::operator ++ () -> Iterator& {
  CHECK_EQ(bfs_queue_.empty(), false);
  NodeType* cur_node = bfs_queue_.front();
  bfs_queue_.pop();
  for (EdgeType* out_edge : cur_node->out_edges()) {
    bfs_queue_.push(out_edge->dst_node());
  }
  return *this;
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::Iterator::operator != (
    const Graph::Iterator& rhs) const {
  if (bfs_queue_.empty() != rhs.bfs_queue_.empty()) {
    return true;
  }
  if (bfs_queue_.empty() == false && rhs.bfs_queue_.empty() == false) {
    return bfs_queue_.front() != rhs.bfs_queue_.front();
  }
  return false;
}

template<typename NodeType, typename EdgeType>
NodeType& Graph<NodeType, EdgeType>::ReverseIterator::operator * () {
  CHECK_EQ(bfs_queue_.empty(), false);
  return *(bfs_queue_.front());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::ReverseIterator::operator -> () {
  return &(*(*this));
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::ReverseIterator::operator ++ () -> ReverseIterator& {
  CHECK_EQ(bfs_queue_.empty(), false);
  NodeType* cur_node = bfs_queue_.front();
  bfs_queue_.pop();
  for (EdgeType* in_edge : cur_node->in_edges()) {
    bfs_queue_.push(in_edge->src_node());
  }
  return *this;
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::ReverseIterator::operator != (
    const Graph::ReverseIterator& rhs) const {
  if (bfs_queue_.empty() != rhs.bfs_queue_.empty()) {
    return true;
  }
  if (bfs_queue_.empty() == false && rhs.bfs_queue_.empty() == false) {
    return bfs_queue_.front() != rhs.bfs_queue_.front();
  }
  return false;
}

} // namespace oneflow

#endif // ONEFLOW_GRAPH_GRAPH_H_
