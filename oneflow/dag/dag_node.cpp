#include "dag_node.h"
#include "glog/logging.h"

namespace oneflow {

// be careful, this implemention is not thread-safe
void DagNode::init(const std::string& name) {
  static int32_t node_id_cnt = 0;
  node_id_ = node_id_cnt++;
  name_ = name;
}

bool DagNode::AddPredecessor(DagNode* predecessor_ptr) {
  bool pred_success = predecessor_ptr->successors_.insert(node_id_).second;
  bool this_success = predecessors_.insert(predecessor_ptr->node_id()).second;
  CHECK_EQ(pred_success, this_success)
    << "Either it has been inserted or not";
  return pred_success;
}

bool DagNode::RemovePredecessor(DagNode* predecessor_ptr) {
  bool pred_success = predecessor_ptr->successors_.erase(node_id_);
  bool this_success = predecessors_.erase(predecessor_ptr->node_id());
  CHECK_EQ(pred_success, this_success)
    << "Either it has been erased or not";
  return pred_success;
}

} // namespace oneflow
