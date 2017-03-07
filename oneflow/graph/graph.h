#ifndef ONEFLOW_GRAPH_GRAPH_H_
#define ONEFLOW_GRAPH_GRAPH_H_

#include <iostream>
#include <queue>
#include "graph/node.h"

namespace oneflow {

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

  void Init() {
    start_node_.Init();
    stop_node_.Init();
  }

  const Node& start_node() const { return start_node_; }
  const Node& stop_node() const { return stop_node_; }

  GraphIterator begin();
  GraphIterator end();
  ConstGraphIterator cbegin() const;
  ConstGraphIterator cend() const;

  ReverseGraphIterator rbegin();
  ReverseGraphIterator rend();
  ConstReverseGraphIterator crbegin() const;
  ConstReverseGraphIterator crend() const;
  
  const std::vector<std::unique_ptr<Node>>& node_vec() const {
    return node_vec_;
  }
  
  bool IsFirstNode(const Node* node) const {
    return start_node_.HasSuccessor(node);
  }
  bool IsLastNode(const Node* node) const {
    return stop_node_.HasPredecessor(node);
  }

 protected:
  void UpdateStartAndStop();

  void RegisterNode(Node* new_node) {
    node_vec_.emplace_back(new_node);
  }
  void RegisterNode(std::unique_ptr<Node>&& new_node) {
    node_vec_.push_back(std::move(new_node));
  }
  void RegisterEdge(Edge* new_edge) {
    edge_vec_.emplace_back(new_edge);
  }

 private:
  Node start_node_;
  Node stop_node_;
  std::vector<std::unique_ptr<Edge>> start_edge_vec_; // edges on start
  std::vector<std::unique_ptr<Edge>> stop_edge_vec_; // edges on stop
  
  // manage the delete of nodes,edges that are not related to start,stop
  std::vector<std::unique_ptr<Node>> node_vec_;
  std::vector<std::unique_ptr<Edge>> edge_vec_;
};

class Graph::GraphIterator {
 public:
  // DISALLOW_MOVE(GraphIterator);
  GraphIterator(const GraphIterator& rhs) { (*this) = rhs; }
  GraphIterator& operator = (const GraphIterator& rhs) {
    if (this != &rhs) {
      bfs_queue_ = std::make_shared<std::queue<Node*>> ();
      *bfs_queue_ = *(rhs.bfs_queue_);
    }
    return *this;
  }
  
  GraphIterator() = default;
  ~GraphIterator() = default;
  
  void Init(Node* start_node) {
    bfs_queue_ = std::make_shared<std::queue<Node*>> ();
    bfs_queue_->push(start_node);
  }
  
  Node& operator * ();
  Node* operator -> ();
  void operator ++ ();
  
  bool operator != (const GraphIterator&) const;

 private:
  // we need to make light-object
  std::shared_ptr<std::queue<Node*>> bfs_queue_;
};

class Graph::ConstGraphIterator {
 public:
  // DISALLOW_COPY_AND_MOVE(ConstGraphIterator);
  ConstGraphIterator() = default;
  ~ConstGraphIterator() = default;
  
  void Init(GraphIterator dag_iterator) {
    dag_iterator_ = dag_iterator;
  }
  
  const Node& operator * () { return *dag_iterator_; }
  const Node* operator -> () { return &(*dag_iterator_); }
  void operator ++ () { ++dag_iterator_; }
  bool operator != (const ConstGraphIterator& rhs) const {
    return dag_iterator_ != rhs.dag_iterator_;
  }

 private:
  GraphIterator dag_iterator_;
};

class Graph::ReverseGraphIterator {
 public:
  // DISALLOW_MOVE(ReverseGraphIterator);
  ReverseGraphIterator(const ReverseGraphIterator& rhs) {
    (*this) = rhs;
  }
  ReverseGraphIterator& operator = (const ReverseGraphIterator& rhs) {
    if (this != &rhs) {
      bfs_queue_ = std::make_shared<std::queue<Node*>> ();
      *bfs_queue_ = *(rhs.bfs_queue_);
    }
    return *this;
  }
  
  ReverseGraphIterator() = default;
  ~ReverseGraphIterator() = default;
  
  void Init(Node* stop_node) {
    bfs_queue_ = std::make_shared<std::queue<Node*>> ();
    bfs_queue_->push(stop_node);
  }
  
  Node& operator * ();
  Node* operator -> ();
  void operator ++ ();
  
  bool operator != (const ReverseGraphIterator&) const;

 private:
  // we need to make light-object
  std::shared_ptr<std::queue<Node*>> bfs_queue_;
};

class Graph::ConstReverseGraphIterator {
 public:
  // DISALLOW_COPY_AND_MOVE(ConstReverseGraphIterator);
  ConstReverseGraphIterator() = default;
  ~ConstReverseGraphIterator() = default;
  
  void Init(ReverseGraphIterator dag_iterator) {
    dag_iterator_ = dag_iterator;
  }
  
  const Node& operator * () { return *dag_iterator_; }
  const Node* operator -> () { return &(*dag_iterator_); }
  void operator ++ () { ++dag_iterator_; }
  bool operator != (const ConstReverseGraphIterator& rhs) const {
    return dag_iterator_ != rhs.dag_iterator_;
  }

 private:
  ReverseGraphIterator dag_iterator_;
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_GRAPH_H_
