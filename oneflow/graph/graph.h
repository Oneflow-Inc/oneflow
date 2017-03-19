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
  class GraphIterator;
  class ConstGraphIterator;
  // Reverse Topologically ergodic all nodes except start_node_,stop_node_
  class ReverseGraphIterator;
  class ConstReverseGraphIterator;

  DISALLOW_COPY_AND_MOVE(Graph);
  Graph() = default;
  virtual ~Graph() = default;

  virtual void Init() {
    start_node_.Init();
    stop_node_.Init();
  }

  const NodeType& start_node() const { return start_node_; }
  const NodeType& stop_node() const { return stop_node_; }

  GraphIterator begin();
  GraphIterator end();
  ConstGraphIterator cbegin() const;
  ConstGraphIterator cend() const;

  ReverseGraphIterator rbegin();
  ReverseGraphIterator rend();
  ConstReverseGraphIterator crbegin() const;
  ConstReverseGraphIterator crend() const;
  
  const std::unordered_set<std::unique_ptr<NodeType>>& nodes() const {
    return nodes_;
  }
  
  bool IsFirstNode(const NodeType* node) const {
    return start_node_.HasSuccessor(node);
  }
  bool IsLastNode(const NodeType* node) const {
    return stop_node_.HasPredecessor(node);
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
  void UnRegisterEdge(EdgeType* old_edge) {
    // It is a high-cost function, should not be called frequently
    for (auto it = edges_.begin(); it != edges_.end(); ++it) {
      if (it->get() == old_edge) {
        edges_.earse(it);
        return;
      }
    }
    LOG(FATAL) << "old edge not found";
  }
  // New
  NodeType* NewFinalNode() {
    // In c++14, we can use std::is_final to check
    NodeType* ret = new NodeType;
    ret->Init();
    RegisterNode(ret);
    return ret;
  }
  EdgeType* NewFinalEdge() {
    // In c++14, we can use std::is_final to check
    EdgeType* ret = new EdgeType;
    ret->Init();
    RegisterEdge(ret);
    return ret;
  }

 private:
  NodeType start_node_;
  NodeType stop_node_;
  std::unordered_set<std::unique_ptr<EdgeType>> start_edges_;
  std::unordered_set<std::unique_ptr<EdgeType>> stop_edges_;
  
  // manage the delete of nodes,edges that are not related to start,stop
  std::unordered_set<std::unique_ptr<NodeType>> nodes_;
  std::unordered_set<std::unique_ptr<EdgeType>> edges_;
};


template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::GraphIterator final {
 public:
  // DISALLOW_MOVE(GraphIterator);
  GraphIterator(const GraphIterator& rhs) { (*this) = rhs; }
  GraphIterator& operator = (const GraphIterator& rhs) {
    if (this != &rhs) {
      bfs_queue_ = std::make_shared<std::queue<NodeType*>> ();
      *bfs_queue_ = *(rhs.bfs_queue_);
    }
    return *this;
  }
  
  GraphIterator() = default;
  ~GraphIterator() = default;
  
  void Init(NodeType* start_node) {
    bfs_queue_ = std::make_shared<std::queue<NodeType*>> ();
    bfs_queue_->push(start_node);
  }
  
  NodeType& operator * ();
  NodeType* operator -> ();
  void operator ++ ();
  
  bool operator != (const GraphIterator&) const;

 private:
  // we need to make light-object
  std::shared_ptr<std::queue<NodeType*>> bfs_queue_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ConstGraphIterator final {
 public:
  // DISALLOW_COPY_AND_MOVE(ConstGraphIterator);
  ConstGraphIterator() = default;
  ~ConstGraphIterator() = default;
  
  void Init(GraphIterator graph_iterator) {
    graph_iterator_ = graph_iterator;
  }
  
  const NodeType& operator * () { return *graph_iterator_; }
  const NodeType* operator -> () { return &(*graph_iterator_); }
  void operator ++ () { ++graph_iterator_; }
  bool operator != (const ConstGraphIterator& rhs) const {
    return graph_iterator_ != rhs.graph_iterator_;
  }

 private:
  GraphIterator graph_iterator_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ReverseGraphIterator final {
 public:
  // DISALLOW_MOVE(ReverseGraphIterator);
  ReverseGraphIterator(const ReverseGraphIterator& rhs) {
    (*this) = rhs;
  }
  ReverseGraphIterator& operator = (const ReverseGraphIterator& rhs) {
    if (this != &rhs) {
      bfs_queue_ = std::make_shared<std::queue<NodeType*>> ();
      *bfs_queue_ = *(rhs.bfs_queue_);
    }
    return *this;
  }
  
  ReverseGraphIterator() = default;
  ~ReverseGraphIterator() = default;
  
  void Init(NodeType* stop_node) {
    bfs_queue_ = std::make_shared<std::queue<NodeType*>> ();
    bfs_queue_->push(stop_node);
  }
  
  NodeType& operator * ();
  NodeType* operator -> ();
  void operator ++ ();
  
  bool operator != (const ReverseGraphIterator&) const;

 private:
  // we need to make light-object
  std::shared_ptr<std::queue<NodeType*>> bfs_queue_;
};

template<typename NodeType, typename EdgeType>
class Graph<NodeType, EdgeType>::ConstReverseGraphIterator final {
 public:
  // DISALLOW_COPY_AND_MOVE(ConstReverseGraphIterator);
  ConstReverseGraphIterator() = default;
  ~ConstReverseGraphIterator() = default;
  
  void Init(ReverseGraphIterator graph_iterator) {
    graph_iterator_ = graph_iterator;
  }
  
  const NodeType& operator * () { return *graph_iterator_; }
  const NodeType* operator -> () { return &(*graph_iterator_); }
  void operator ++ () { ++graph_iterator_; }
  bool operator != (const ConstReverseGraphIterator& rhs) const {
    return graph_iterator_ != rhs.graph_iterator_;
  }

 private:
  ReverseGraphIterator graph_iterator_;
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
auto Graph<NodeType, EdgeType>::begin() -> GraphIterator {
  GraphIterator ret;
  ret.Init(&start_node_);
  ++ret;
  return ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::end() -> GraphIterator {
  GraphIterator ret;
  ret.Init(&stop_node_);
  return ret;
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::cbegin() const -> ConstGraphIterator{
  ConstGraphIterator ret;
  ret.Init((const_cast<Graph*>(this))->begin());
  return ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::cend() const -> ConstGraphIterator {
  ConstGraphIterator ret;
  ret.Init((const_cast<Graph*>(this))->end());
  return ret;
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::rbegin() -> ReverseGraphIterator {
  ReverseGraphIterator ret;
  ret.Init(&stop_node_);
  ++ret;
  return ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::rend() -> ReverseGraphIterator {
  ReverseGraphIterator ret;
  ret.Init(&start_node_);
  return ret;
}

template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::crbegin() const -> ConstReverseGraphIterator {
  ConstReverseGraphIterator ret;
  ret.Init((const_cast<Graph*>(this))->rbegin());
  return ret;
}
template<typename NodeType, typename EdgeType>
auto Graph<NodeType, EdgeType>::crend() const -> ConstReverseGraphIterator {
  ConstReverseGraphIterator ret;
  ret.Init((const_cast<Graph*>(this))->rend());
  return ret;
}

template<typename NodeType, typename EdgeType>
NodeType& Graph<NodeType, EdgeType>::GraphIterator::operator * () {
  CHECK_EQ(bfs_queue_->empty(), false);
  return *(bfs_queue_->front());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::GraphIterator::operator -> () {
  return &(*(*this));
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::GraphIterator::operator ++ () {
  CHECK_EQ(bfs_queue_->empty(), false);
  NodeType* cur_node = bfs_queue_->front();
  bfs_queue_->pop();
  for (EdgeType* out_edge : cur_node->out_edges()) {
    bfs_queue_->push(out_edge->dst_node());
  }
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::GraphIterator::operator != (
    const Graph::GraphIterator& rhs) const {
  if (bfs_queue_->empty() != rhs.bfs_queue_->empty()) {
    return true;
  }
  if (bfs_queue_->empty() == false && rhs.bfs_queue_->empty() == false) {
    return bfs_queue_->front() != rhs.bfs_queue_->front();
  }
  return false;
}

template<typename NodeType, typename EdgeType>
NodeType& Graph<NodeType, EdgeType>::ReverseGraphIterator::operator * () {
  CHECK_EQ(bfs_queue_->empty(), false);
  return *(bfs_queue_->front());
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::ReverseGraphIterator::operator -> () {
  return &(*(*this));
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ReverseGraphIterator::operator ++ () {
  CHECK_EQ(bfs_queue_->empty(), false);
  NodeType* cur_node = bfs_queue_->front();
  bfs_queue_->pop();
  for (EdgeType* in_edge : cur_node->in_edges()) {
    bfs_queue_->push(in_edge->src_node());
  }
}

template<typename NodeType, typename EdgeType>
bool Graph<NodeType, EdgeType>::ReverseGraphIterator::operator != (
    const Graph::ReverseGraphIterator& rhs) const {
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
