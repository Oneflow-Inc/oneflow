#include "dag/dag.h"
#include "glog/logging.h"

namespace oneflow {

Dag::DagIterator::DagIterator(const Dag::DagIterator& rhs) {
  (*this) = rhs;
}

Dag::DagIterator& Dag::DagIterator::operator = (const Dag::DagIterator& rhs) {
  if (this != &rhs) {
    bfs_queue_ = std::make_shared<std::queue<DagNode*>> ();
    *bfs_queue_ = *(rhs.bfs_queue_);
  }
  return *this;
}

DagNode& Dag::DagIterator::operator * () {
  CHECK_EQ(bfs_queue_->empty(), false);
  return *(bfs_queue_->front());
}

DagNode* Dag::DagIterator::operator -> () {
  return &(*(*this));
}

void Dag::DagIterator::operator ++ () {
  CHECK_EQ(bfs_queue_->empty(), false);
  DagNode* cur_node = bfs_queue_->front();
  bfs_queue_->pop();
  for (DagNode* successor : cur_node->successors()) {
    bfs_queue_->push(successor);
  }
}

bool Dag::DagIterator::operator != (const Dag::DagIterator& rhs) const {
  if (bfs_queue_->empty() != rhs.bfs_queue_->empty()) {
    return true;
  }
  if (bfs_queue_->empty() == false && rhs.bfs_queue_->empty() == false) {
    return bfs_queue_->front() != rhs.bfs_queue_->front();
  }
  return false;
}

Dag::ReverseDagIterator::ReverseDagIterator(const Dag::ReverseDagIterator& rhs) {
  (*this) = rhs;
}

Dag::ReverseDagIterator& Dag::ReverseDagIterator::operator = (const Dag::ReverseDagIterator& rhs) {
  if (this != &rhs) {
    bfs_queue_ = std::make_shared<std::queue<DagNode*>> ();
    *bfs_queue_ = *(rhs.bfs_queue_);
  }
  return *this;
}

DagNode& Dag::ReverseDagIterator::operator * () {
  CHECK_EQ(bfs_queue_->empty(), false);
  return *(bfs_queue_->front());
}

DagNode* Dag::ReverseDagIterator::operator -> () {
  return &(*(*this));
}

void Dag::ReverseDagIterator::operator ++ () {
  CHECK_EQ(bfs_queue_->empty(), false);
  DagNode* cur_node = bfs_queue_->front();
  bfs_queue_->pop();
  for (DagNode* predecessor : cur_node->predecessors()) {
    bfs_queue_->push(predecessor);
  }
}

bool Dag::ReverseDagIterator::operator != (const Dag::ReverseDagIterator& rhs) const {
  if (bfs_queue_->empty() != rhs.bfs_queue_->empty()) {
    return true;
  }
  if (bfs_queue_->empty() == false && rhs.bfs_queue_->empty() == false) {
    return bfs_queue_->front() != rhs.bfs_queue_->front();
  }
  return false;
}

void Dag::ConnectStartAndStop() {
  for (const std::unique_ptr<DagNode>& node : data_op_node_vec_) {
    if (node->predecessors().empty()) {
      node->AddPredecessor(&start_node_);
    }
    if (node->successors().empty()) {
      stop_node_.AddPredecessor(node.get());
    }
  }
}

void Dag::ConnectOpNodeExtraPtr() {
  for (OpNode* cur_node : op_node_vec_) {
    for (DagNode* data_pre_node : cur_node->predecessors()) {
      for (DagNode* op_pre_node : data_pre_node->predecessors()) {
        cur_node->mutable_op_predecessors().insert(
            of_dynamic_cast<OpNode*> (op_pre_node));
      }
    }
    for (DagNode* data_next_node : cur_node->successors()) {
      for (DagNode* op_next_node : data_next_node->successors()) {
        cur_node->mutable_op_successors().insert(
            of_dynamic_cast<OpNode*> (op_next_node));
      }
    }
  }
}

} // namespace oneflow
