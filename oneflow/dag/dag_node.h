#ifndef ONEFLOW_DAG_DAG_NODE_H_
#define ONEFLOW_DAG_DAG_NODE_H_

#include <stdint.h>
#include <unordered_set>
#include <set>
#include <vector>
#include <memory>

namespace oneflow {

class DagNode {
 public:
  DagNode() = default;
  DagNode(const DagNode&) = delete;
  DagNode(DagNode&&) = delete;
  DagNode& operator = (const DagNode&) = delete;
  DagNode& operator = (DagNode&&) = delete;
  virtual ~DagNode() = default;

  void init(const std::string& name);
  
  int32_t node_id() const { return node_id_; }
  const std::string& node_name() const { return node_name_; }

  // return false if it already existed
  bool AddPredecessor(DagNode* predecessor_ptr);
  bool RemovePredecessor(DagNode* predecessor_ptr);

  const std::set<int32_t>& predecessors() const { return predecessors_; }
  const std::set<int32_t>& successors() const { return successors_; }

 private:
  const int32_t node_id_;
  std::string node_name_;
  
  // Use std::set instead of std::unordered_set to keep the increasing
  // order of node_id while traversing the DAG
  std::set<int32_t> predecessors_;
  std::set<int32_t> successors_;

};

template <typename Data>
class DataNode : public DagNode {
 public:
  DataNode() = default;
  DataNode(const DataNode&) = delete;
  DataNode(DataNode&&) = delete;
  DataNode& operator = (const DataNode&) = delete;
  DataNode& operator = (DataNode&&) = delete;
  ~DataNode() = default;

  void init(const std::string& name, const std::shared_ptr<Data>& data) {
    Base::init(name);
    data_ = data;
  }

  std::shared_ptr<const Data>& data() const { return data_; }
  std::shared_ptr<Data>& mutable_data(); { return data_; }

 private:
  std::shared_ptr<Data> data_;
};

template <typename Op>
class OpNode : public DagNode {
 public:
  OpNode() = default;
  OpNode(const OpNode&) = delete;
  OpNode(OpNode&&) = delete;
  OpNode& operator = (const OpNode&) = delete;
  OpNode& operator = (OpNode&&) = delete;
  ~OpNode() = default;

  void init(const std::string& name, const std::shared_ptr<Op>& op) {
    Base::init(name);
    op_ = op;
  }

  std::shared_ptr<const Op>& op() const { return op_; }
  std::shared_ptr<Op>& mutable_op() { return op_; }
 
 private:
  std::shared_ptr<Op> op_;

};

} // namespace oneflow
#endif  // ONEFLOW_DAG_DAG_NODE_H_
