#ifndef ONEFLOW_GRAPH_GRAPH_H_
#define ONEFLOW_GRAPH_GRAPH_H_

#include <iostream>
#include <queue>
#include "graph/node.h"

namespace oneflow {

class Graph {
 public:
  // Topologically ergodic all nodes except start_node_,stop_node_
  class GraphIterator {
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
  class ConstGraphIterator {
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

  // Reverse Topologically ergodic all nodes except start_node_,stop_node_
  class ReverseGraphIterator {
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
  class ConstReverseGraphIterator {
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

  DISALLOW_COPY_AND_MOVE(Graph);
  Graph() = default;
  virtual ~Graph() = default;

  void Init() {
    start_node_.Init();
    stop_node_.Init();
  }

  const Node& start_node() const { return start_node_; }
  const Node& stop_node() const { return stop_node_; }

  GraphIterator begin() {
    GraphIterator ret;
    ret.Init(&start_node_);
    ++ret;
    return ret;
  }
  GraphIterator end() {
    GraphIterator ret;
    ret.Init(&stop_node_);
    return ret;
  }
  ConstGraphIterator cbegin() const {
    ConstGraphIterator ret;
    ret.Init((const_cast<Graph*>(this))->begin());
    return ret;
  }
  ConstGraphIterator cend() const {
    ConstGraphIterator ret;
    ret.Init((const_cast<Graph*>(this))->end());
    return ret;
  }

  ReverseGraphIterator rbegin() {
    ReverseGraphIterator ret;
    ret.Init(&stop_node_);
    ++ret;
    return ret;
  }
  ReverseGraphIterator rend() {
    ReverseGraphIterator ret;
    ret.Init(&start_node_);
    return ret;
  }
  ConstReverseGraphIterator crbegin() const {
    ConstReverseGraphIterator ret;
    ret.Init((const_cast<Graph*>(this))->rbegin());
    return ret;
  }
  ConstReverseGraphIterator crend() const {
    ConstReverseGraphIterator ret;
    ret.Init((const_cast<Graph*>(this))->rend());
    return ret;
  }
  
  const std::vector<std::unique_ptr<Node>>& node_vec() const {
    return node_vec_;
  }
  
  bool IsFirstNode(const Node* node) const {
    auto find_it = start_node_.successors().find(const_cast<Node*> (node));
    return find_it != start_node_.successors().end();
  }
  bool IsLastNode(const Node* node) const {
    auto find_it = stop_node_.predecessors().find(const_cast<Node*> (node));
    return find_it != stop_node_.predecessors().end();
  }

 protected:
  void UpdateStartAndStop();

  void RegisterNode(Node* new_node) {
    node_vec_.emplace_back(new_node);
  }

 private:
  Node start_node_;
  Node stop_node_;

  std::vector<std::unique_ptr<Node>> node_vec_; 

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_GRAPH_H_
