#include "graph/graph.h"
#include "glog/logging.h"

namespace oneflow {

Node& Graph::GraphIterator::operator * () {
  CHECK_EQ(bfs_queue_->empty(), false);
  return *(bfs_queue_->front());
}

Node* Graph::GraphIterator::operator -> () {
  return &(*(*this));
}

void Graph::GraphIterator::operator ++ () {
  CHECK_EQ(bfs_queue_->empty(), false);
  Node* cur_node = bfs_queue_->front();
  bfs_queue_->pop();
  for (Node* successor : cur_node->successors()) {
    bfs_queue_->push(successor);
  }
}

bool Graph::GraphIterator::operator != (
    const Graph::GraphIterator& rhs) const {
  if (bfs_queue_->empty() != rhs.bfs_queue_->empty()) {
    return true;
  }
  if (bfs_queue_->empty() == false && rhs.bfs_queue_->empty() == false) {
    return bfs_queue_->front() != rhs.bfs_queue_->front();
  }
  return false;
}

Node& Graph::ReverseGraphIterator::operator * () {
  CHECK_EQ(bfs_queue_->empty(), false);
  return *(bfs_queue_->front());
}

Node* Graph::ReverseGraphIterator::operator -> () {
  return &(*(*this));
}

void Graph::ReverseGraphIterator::operator ++ () {
  CHECK_EQ(bfs_queue_->empty(), false);
  Node* cur_node = bfs_queue_->front();
  bfs_queue_->pop();
  for (Node* predecessor : cur_node->predecessors()) {
    bfs_queue_->push(predecessor);
  }
}

bool Graph::ReverseGraphIterator::operator != (
    const Graph::ReverseGraphIterator& rhs) const {
  if (bfs_queue_->empty() != rhs.bfs_queue_->empty()) {
    return true;
  }
  if (bfs_queue_->empty() == false && rhs.bfs_queue_->empty() == false) {
    return bfs_queue_->front() != rhs.bfs_queue_->front();
  }
  return false;
}

void Graph::UpdateStartAndStop() {
  start_node_.clear_predecessors();
  start_node_.clear_successors();
  stop_node_.clear_predecessors();
  stop_node_.clear_successors();
  for (const std::unique_ptr<Node>& node : node_vec_) {
    if (node->predecessors().empty()) {
      ConnectTwoNode(&start_node_, node.get());
    }
    if (node->successors().empty()) {
      ConnectTwoNode(node.get(), &stop_node_);
    }
  }
}

} // namespace oneflow
