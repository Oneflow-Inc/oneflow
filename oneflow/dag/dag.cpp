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
}

} // namespace oneflow
