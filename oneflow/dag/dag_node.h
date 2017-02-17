#ifndef ONEFLOW_DAG_DAG_NODE_H_
#define ONEFLOW_DAG_DAG_NODE_H_

#include <stdint.h>
#include <unordered_set>
#include <set>
#include <vector>
#include <memory>
#include "common/util.h"

namespace oneflow {

class DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(DagNode);

  int32_t node_id() const { return node_id_; }

  // return false if it has already been inserted
  bool AddPredecessor(DagNode* predecessor_ptr);
  // return false if it has already been erased
  bool RemovePredecessor(DagNode* predecessor_ptr);

  const std::set<int32_t>& predecessors() const { return predecessors_; }
  const std::set<int32_t>& successors() const { return successors_; }

 protected:
  DagNode() = default;
  virtual ~DagNode() = default;
  
  void init();

 private:
  int32_t node_id_;
  
  // Use std::set instead of std::unordered_set to keep the increasing
  // order of node_id while traversing the DAG
  std::set<int32_t> predecessors_;
  std::set<int32_t> successors_;

};

template <typename Data>
class DataNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(DataNode);

  DataNode() = default;
  ~DataNode() = default;

  void init(const std::shared_ptr<Data>& data) {
    DagNode::init();
    data_ = data;
  }

  std::shared_ptr<const Data>& data() const { return data_; }
  std::shared_ptr<Data>& mutable_data() { return data_; }

 private:
  std::shared_ptr<Data> data_;
};

template <typename Op>
class OpNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(OpNode);

  OpNode() = default;
  ~OpNode() = default;

  void init(const std::shared_ptr<Op>& op) {
    DagNode::init();
    op_ = op;
  }

  std::shared_ptr<const Op>& op() const { return op_; }
  std::shared_ptr<Op>& mutable_op() { return op_; }
 
 private:
  std::shared_ptr<Op> op_;

};

} // namespace oneflow
#endif  // ONEFLOW_DAG_DAG_NODE_H_
