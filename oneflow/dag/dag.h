#ifndef ONEFLOW_DAG_DAG_H_
#define ONEFLOW_DAG_DAG_H_

#include <iostream>
#include <queue>
#include "dag/dag_node.h"

namespace oneflow {

class Dag {
 public:
  // Topologically ergodic all nodes except start_node_,stop_node_
  class DagIterator {
   public:
    // DISALLOW_MOVE(DagIterator);
    DagIterator(const DagIterator&);
    DagIterator& operator = (const DagIterator&);
    
    DagIterator() = default;
    ~DagIterator() = default;
    
    void Init(DagNode* start_node) {
      bfs_queue_ = std::make_shared<std::queue<DagNode*>> ();
      bfs_queue_->push(start_node);
    }
    
    DagNode& operator * ();
    DagNode* operator -> ();
    void operator ++ ();
    
    bool operator != (const DagIterator&) const;

   private:
    // we need to make light-object
    std::shared_ptr<std::queue<DagNode*>> bfs_queue_;
  };
  class ConstDagIterator {
   public:
    // DISALLOW_COPY_AND_MOVE(ConstDagIterator);
    ConstDagIterator() = default;
    ~ConstDagIterator() = default;
    
    void Init(DagIterator dag_iterator) {
      dag_iterator_ = dag_iterator;
    }
    
    const DagNode& operator * () { return *dag_iterator_; }
    const DagNode* operator -> () { return &(*dag_iterator_); }
    void operator ++ () { ++dag_iterator_; }
    bool operator != (const ConstDagIterator& rhs) const {
      return dag_iterator_ != rhs.dag_iterator_;
    }

   private:
    DagIterator dag_iterator_;
  };

  // Reverse Topologically ergodic all nodes except start_node_,stop_node_
  class ReverseDagIterator {
   public:
    // DISALLOW_MOVE(ReverseDagIterator);
    ReverseDagIterator(const ReverseDagIterator&);
    ReverseDagIterator& operator = (const ReverseDagIterator&);
    
    ReverseDagIterator() = default;
    ~ReverseDagIterator() = default;
    
    void Init(DagNode* stop_node) {
      bfs_queue_ = std::make_shared<std::queue<DagNode*>> ();
      bfs_queue_->push(stop_node);
    }
    
    DagNode& operator * ();
    DagNode* operator -> ();
    void operator ++ ();
    
    bool operator != (const ReverseDagIterator&) const;

   private:
    // we need to make light-object
    std::shared_ptr<std::queue<DagNode*>> bfs_queue_;
  };
  class ConstReverseDagIterator {
   public:
    // DISALLOW_COPY_AND_MOVE(ConstReverseDagIterator);
    ConstReverseDagIterator() = default;
    ~ConstReverseDagIterator() = default;
    
    void Init(ReverseDagIterator dag_iterator) {
      dag_iterator_ = dag_iterator;
    }
    
    const DagNode& operator * () { return *dag_iterator_; }
    const DagNode* operator -> () { return &(*dag_iterator_); }
    void operator ++ () { ++dag_iterator_; }
    bool operator != (const ConstReverseDagIterator& rhs) const {
      return dag_iterator_ != rhs.dag_iterator_;
    }

   private:
    ReverseDagIterator dag_iterator_;
  };

  DISALLOW_COPY_AND_MOVE(Dag);
  Dag() = default;
  virtual ~Dag() = default;

  void Init(const std::string& dag_name) {
    dag_name_ = dag_name;
    start_node_.Init();
    stop_node_.Init();
  }

  const std::string& dag_name() { return dag_name_; }
  const DagNode& start_node() const { return start_node_; }
  const DagNode& stop_node() const { return stop_node_; }

  DagIterator begin() {
    DagIterator ret;
    ret.Init(&start_node_);
    ++ret;
    return ret;
  }
  DagIterator end() {
    DagIterator ret;
    ret.Init(&stop_node_);
    return ret;
  }
  ConstDagIterator cbegin() const {
    ConstDagIterator ret;
    ret.Init((const_cast<Dag*>(this))->begin());
    return ret;
  }
  ConstDagIterator cend() const {
    ConstDagIterator ret;
    ret.Init((const_cast<Dag*>(this))->end());
    return ret;
  }

  ReverseDagIterator rbegin() {
    ReverseDagIterator ret;
    ret.Init(&stop_node_);
    ++ret;
    return ret;
  }
  ReverseDagIterator rend() {
    ReverseDagIterator ret;
    ret.Init(&start_node_);
    return ret;
  }
  ConstReverseDagIterator crbegin() const {
    ConstReverseDagIterator ret;
    ret.Init((const_cast<Dag*>(this))->rbegin());
    return ret;
  }
  ConstReverseDagIterator crend() const {
    ConstReverseDagIterator ret;
    ret.Init((const_cast<Dag*>(this))->rend());
    return ret;
  }
  
  const std::vector<OpNode*>& op_node_vec() const {
    return op_node_vec_;
  }
  
  bool IsFirstNode(const DagNode* node) const {
    auto find_it = start_node_.successors().find(const_cast<DagNode*> (node));
    return find_it != start_node_.successors().end();
  }
  bool IsLastNode(const DagNode* node) const {
    auto find_it = stop_node_.predecessors().find(const_cast<DagNode*> (node));
    return find_it != stop_node_.predecessors().end();
  }

 protected:
  void ConnectStartAndStop();
  void ConnectOpNodeExtraPtr();

  void RegisterDataNode(std::unique_ptr<DataNode> new_node) {
    data_node_vec_.push_back(new_node.get());
    data_op_node_vec_.push_back(std::move(new_node));
  }
  void RegisterOpNode(std::unique_ptr<OpNode> new_node) {
    op_node_vec_.push_back(new_node.get());
    data_op_node_vec_.push_back(std::move(new_node));
  }

 private:
  std::string dag_name_;
  DagNode start_node_;
  DagNode stop_node_;

  // In future we can implement a Iterator to replace the data_op_node_vec_
  // which is redundancy
  std::vector<std::unique_ptr<DagNode>> data_op_node_vec_; 
  std::vector<DataNode*> data_node_vec_;
  std::vector<OpNode*> op_node_vec_;

};

} // namespace oneflow

#endif // ONEFLOW_DAG_DAG_H_
