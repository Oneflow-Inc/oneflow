#ifndef ONEFLOW_GRAPH_NODE_H_
#define ONEFLOW_GRAPH_NODE_H_

#include <stdint.h>
#include <unordered_set>
#include <vector>
#include <memory>
#include "common/util.h"

namespace oneflow {

class Node {
 public:
  DISALLOW_COPY_AND_MOVE(Node);
  virtual ~Node() = default;
  
  Node() = default;
  void Init();

  int32_t node_id() const { return node_id_; }

  const std::unordered_set<Node*>& predecessors() const {
    return predecessors_;
  }
  const std::unordered_set<Node*>& successors() const {
    return successors_;
  }

 private:
  friend bool ConnectTwoNode(Node* predecessor, Node* successor);

  int32_t node_id_;
  
  std::unordered_set<Node*> predecessors_;
  std::unordered_set<Node*> successors_;

};

} // namespace oneflow

#endif  // ONEFLOW_GRAPH_NODE_H_
