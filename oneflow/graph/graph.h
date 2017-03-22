#ifndef ONEFLOW_GRAPH_GRAPH_H_
#define ONEFLOW_GRAPH_GRAPH_H_

#include <iostream>
#include <queue>
#include "graph/node.h"

namespace oneflow {

template<typename NodeType, typename EdgeType>
class Graph {
 public:
  // Topologically ergodic all nodes except start_node_,stop_node_
  class Iterator;
  class ConstIterator;
  // Reverse Topologically ergodic all nodes except start_node_,stop_node_
  class ReverseIterator;
  class ConstReverseIterator;

  OF_DISALLOW_COPY_AND_MOVE(Graph);
  Graph() = default;
  virtual ~Graph() = default;

  const NodeType& start_node() const { return start_node_; }
  const NodeType& stop_node() const { return stop_node_; }

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
  
  const std::unordered_set<std::unique_ptr<NodeType>>& nodes() const {
    return nodes_;
  }
  const std::unordered_set<std::unique_ptr<EdgeType>>& edges() const {
    return edges_;
  }
  
  bool IsFirstNode(const NodeType* node) const {
    return node->SoleInEdge()->src_node() == &start_node_;
  }
  bool IsLastNode(const NodeType* node) const {
    return node->SoleOutEdge()->dst_node() == &stop_node_;
  }

 protected:
  void UpdateStartAndStop();

  // Register
  void RegisterNode(NodeType* new_node) {
    nodes_.emplace(new_node);
  }
  void RegisterNode(std::unique_ptr<NodeType>&& new_node) {
    nodes_.insert(std::move(new_node));
  }
  void RegisterEdge(EdgeType* new_edge) {
    edges_.emplace(new_edge);
  }
  void RegisterEdge(std::unique_ptr<EdgeType>&& new_node) {
    edges_.insert(std::move(new_node));
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
  NodeType start_node_;
  NodeType stop_node_;
  std::vector<std::unique_ptr<EdgeType>> start_edges_;
  std::vector<std::unique_ptr<EdgeType>> stop_edges_;
  
  // manage nodes,edges that are not related to start,stop
  std::vector<std::unique_ptr<NodeType>> nodes_;
  std::vector<std::unique_ptr<EdgeType>> edges_;
};


template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::Iterator final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Iterator);
  Iterator() = default;
  ~Iterator() = default;
  
  void Init(NodeType* start_node) {
    bfs_queue_.clear();
    bfs_queue_.push(start_node);
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
  
  void Init(NodeType* stop_node) {
    bfs_queue_.clear();
    bfs_queue_.push(stop_node);
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
void Graph<NodeType, EdgeType>::UpdateStartAndStop() {
  start_node_.DisconnectAllEdges();
  stop_node_.DisconnectAllEdges();
  start_edges_.clear();
  stop_edges_.clear();
  for (const std::unique_ptr<NodeType>& node : nodes_) {
    if (node->in_edges().empty()) {
      EdgeType* start_edge = new EdgeType;
      start_edges_.emplace(start_edge);
      start_edge->Init();
      Connect(&start_node_, start_edge, node.get());
    }
    if (node->out_edges().empty()) {
      EdgeType* stop_edge = new EdgeType;
      stop_edges_.emplace(stop_edge);
      stop_edge->Init();
      Connect(node.get(), stop_edge, &stop_node_);
    }
  }
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::begin() -> Iterator {
  Iterator ret;
  ret.Init(&start_node_);
  return ++ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::end() -> Iterator {
  Iterator ret;
  ret.Init(&stop_node_);
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
  ret.Init(&stop_node_);
  return ++ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::rend() -> ReverseIterator {
  ReverseIterator ret;
  ret.Init(&start_node_);
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
  CHECK_EQ(bfs_queue_->empty(), false);
  return *(bfs_queue_->front());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::Iterator::operator -> () {
  return &(*(*this));
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::Iterator::operator ++ () -> Iterator& {
  CHECK_EQ(bfs_queue_->empty(), false);
  NodeType* cur_node = bfs_queue_->front();
  bfs_queue_->pop();
  for (EdgeType* out_edge : cur_node->out_edges()) {
    bfs_queue_->push(out_edge->dst_node());
  }
  return *this;
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::Iterator::operator != (
    const Graph::Iterator& rhs) const {
  if (bfs_queue_->empty() != rhs.bfs_queue_->empty()) {
    return true;
  }
  if (bfs_queue_->empty() == false && rhs.bfs_queue_->empty() == false) {
    return bfs_queue_->front() != rhs.bfs_queue_->front();
  }
  return false;
}

template<typename NodeType, typename EdgeType>
NodeType& Graph<NodeType, EdgeType>::ReverseIterator::operator * () {
  CHECK_EQ(bfs_queue_->empty(), false);
  return *(bfs_queue_->front());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::ReverseIterator::operator -> () {
  return &(*(*this));
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::ReverseIterator::operator ++ () -> ReverseIterator& {
  CHECK_EQ(bfs_queue_->empty(), false);
  NodeType* cur_node = bfs_queue_->front();
  bfs_queue_->pop();
  for (EdgeType* in_edge : cur_node->in_edges()) {
    bfs_queue_->push(in_edge->src_node());
  }
  return *this;
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::ReverseIterator::operator != (
    const Graph::ReverseIterator& rhs) const {
  if (bfs_queue_->empty() != rhs.bfs_queue_->empty()) {
    return true;
  }
  if (bfs_queue_->empty() == false && rhs.bfs_queue_->empty() == false) {
    return bfs_queue_->front() != rhs.bfs_queue_->front();
  }
  return false;
}

} // namespace oneflow

#endif // ONEFLOW_GRAPH_GRAPH_H_
