#ifndef ONEFLOW_DAG_DAG_NODE_H_
#define ONEFLOW_DAG_DAG_NODE_H_

#include <stdint.h>
#include <unordered_set>
#include <vector>
#include <memory>
#include "common/util.h"
#include "dag/data_meta.h"
#include "dag/op_meta.h"

namespace oneflow {

class DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(DagNode);
  virtual ~DagNode() = default;
  
  DagNode() = default;
  void init();

  int32_t node_id() const { return node_id_; }

  // return false if it has already been inserted
  bool AddPredecessor(DagNode* predecessor_ptr);
  // return false if it has already been erased
  bool RemovePredecessor(DagNode* predecessor_ptr);

  const std::unordered_set<DagNode*>& predecessors() const {
    return predecessors_;
  }
  const std::unordered_set<DagNode*>& successors() const {
    return successors_;
  }

 private:
  int32_t node_id_;
  
  std::unordered_set<DagNode*> predecessors_;
  std::unordered_set<DagNode*> successors_;

};

} // namespace oneflow
#endif  // ONEFLOW_DAG_DAG_NODE_H_
