#ifndef _DAG_DAG_NODE_H_
#define _DAG_DAG_NODE_H_
#include <cstdint>
#include <unordered_set>
#include <set>
#include <vector>
#include <memory>
#include <glog/logging.h>

namespace oneflow {

enum class NodeType {
  kUnknown = 0,
  kOpNode,
  kDataNode
};

class DagNode {
 public:
  DagNode(int32_t id, const std::string& name,
    NodeType type = NodeType::kUnknown)
    : node_id_(id), node_name_(name), node_type_(type) {}
  virtual ~DagNode() = default;

  inline int32_t AddParent(DagNode *p);
  inline int32_t RemoveParent(DagNode *p);

  inline int32_t node_id() const { return node_id_;}
  inline const std::string& node_name() const { return node_name_; }
  inline NodeType Type() const { return node_type_; }

  inline const std::set<int32_t>& successors() const;
  inline std::set<int32_t>& mutable_successors();
  inline const std::set<int32_t>& predecessors() const;
  inline std::set<int32_t>& mutable_predecessors();
 protected:
  const int32_t node_id_;
  NodeType node_type_;
  std::string node_name_;
  // Use set<node_id> instead of unordered_set<node_id> to keep the increasing
  // order of node_id while traversing the DAG
  std::set<int32_t> successors_;
  std::set<int32_t> predecessors_;

  DagNode(const DagNode& other) = delete;
  DagNode& operator=(const DagNode& other) = delete;
};

inline int32_t DagNode::AddParent(DagNode* p) {
  int32_t parent_id = p->node_id();
  auto pred_insert_success = p->successors_.insert(node_id_).second;
  auto this_insert_success = predecessors_.insert(parent_id).second;
  // It has already been inserted, or not
  CHECK_EQ(pred_insert_success, this_insert_success)
    << "Either it has been inserted or not";
  return pred_insert_success;
}

inline int32_t DagNode::RemoveParent(DagNode* parent) {
  int32_t parent_id = parent->node_id();
  auto pred_erase_success = parent->successors_.erase(node_id_);
  auto this_erase_success = predecessors_.erase(parent_id);
  CHECK_EQ(pred_erase_success, this_erase_success)
    << "Either it has been erased or not";
  return pred_erase_success;
}

inline const std::set<int32_t>& DagNode::successors() const {
return successors_;
}

inline std::set<int32_t>& DagNode::mutable_successors() {
return successors_;
}

inline const std::set<int32_t>& DagNode::predecessors() const {
return predecessors_;
}

inline std::set<int32_t>& DagNode::mutable_predecessors() {
return predecessors_;
}

template <typename Data>
class DataNode : public DagNode {
 public:
  DataNode(int32_t id, const std::string& name)
    : DagNode(id, name, NodeType::kDataNode) {}
  ~DataNode() = default;
  const std::shared_ptr<Data>& data() const;
  std::shared_ptr<Data>& mutable_data();

 private:
  std::shared_ptr<Data> data_;
  DataNode(const DataNode& other) = delete;
  DataNode& operator=(const DataNode& other) = delete;
};

template <typename Op>
class OpNode : public DagNode {
 public:
  OpNode(int32_t id, const std::string& name)
    : DagNode(id, name, NodeType::kOpNode) {}
  ~OpNode() = default;
  const std::shared_ptr<Op>& op() const;
  std::shared_ptr<Op>& mutable_op();
 private:
  std::shared_ptr<Op> op_;
  OpNode(const OpNode& other) = delete;
  OpNode& operator=(const OpNode& other) = delete;
};

template <typename Data>
inline const std::shared_ptr<Data>& DataNode<Data>::data() const {
  return data_;
}

template <typename Data>
inline std::shared_ptr<Data>& DataNode<Data>::mutable_data() {
  return data_;
}

template <typename Op>
inline const std::shared_ptr<Op>& OpNode<Op>::op() const {
  return op_;
}

template <typename Op>
inline std::shared_ptr<Op>& OpNode<Op>::mutable_op() {
  return op_;
}

}  // namespace oneflow
#endif  // _DAG_DAG_NODE_H_
