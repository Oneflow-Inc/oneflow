#include "dag_node.h"
#include "glog/logging.h"

namespace oneflow {

void DagNode::Init() {
  static int32_t node_id_cnt = 0;
  node_id_ = node_id_cnt++;
}

bool ConnectTwoNode(DagNode* predecessor, DagNode* successor) {
  bool pred_success = predecessor->successors_.insert(successor).second;
  bool succ_success = successor->predecessors_.insert(predecessor).second;
  CHECK_EQ(pred_success, succ_success)
    << "Either it has been inserted or not";
  return pred_success;
}

} // namespace oneflow
