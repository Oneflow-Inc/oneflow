#include "graph/graph.h"
#include "glog/logging.h"

namespace oneflow {

Graph::GraphIterator Graph::begin() {
  GraphIterator ret;
  ret.Init(&start_node_);
  ++ret;
  return ret;
}
Graph::GraphIterator Graph::end() {
  GraphIterator ret;
  ret.Init(&stop_node_);
  return ret;
}

Graph::ConstGraphIterator Graph::cbegin() const {
  ConstGraphIterator ret;
  ret.Init((const_cast<Graph*>(this))->begin());
  return ret;
}
Graph::ConstGraphIterator Graph::cend() const {
  ConstGraphIterator ret;
  ret.Init((const_cast<Graph*>(this))->end());
  return ret;
}

Graph::ReverseGraphIterator Graph::rbegin() {
  ReverseGraphIterator ret;
  ret.Init(&stop_node_);
  ++ret;
  return ret;
}
Graph::ReverseGraphIterator Graph::rend() {
  ReverseGraphIterator ret;
  ret.Init(&start_node_);
  return ret;
}

Graph::ConstReverseGraphIterator Graph::crbegin() const {
  ConstReverseGraphIterator ret;
  ret.Init((const_cast<Graph*>(this))->rbegin());
  return ret;
}
Graph::ConstReverseGraphIterator Graph::crend() const {
  ConstReverseGraphIterator ret;
  ret.Init((const_cast<Graph*>(this))->rend());
  return ret;
}

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
  for (Edge* out_edge : cur_node->out_edges()) {
    bfs_queue_->push(out_edge->dst_node());
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
  for (Edge* in_edge : cur_node->in_edges()) {
    bfs_queue_->push(in_edge->src_node());
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
  start_node_.DisconnectAllEdges();
  stop_node_.DisconnectAllEdges();
  start_edge_vec_.clear();
  stop_edge_vec_.clear();
  for (const std::unique_ptr<Node>& node : node_vec_) {
    if (node->IsEmptyIn()) {
      Edge* start_edge = new Edge;
      start_edge_vec_.emplace_back(start_edge);
      start_edge->Init();
      Connect(&start_node_, start_edge, node.get());
    }
    if (node->IsEmptyOut()) {
      Edge* stop_edge = new Edge;
      stop_edge_vec_.emplace_back(stop_edge);
      stop_edge->Init();
      Connect(node.get(), stop_edge, &stop_node_);
    }
  }
}

} // namespace oneflow
