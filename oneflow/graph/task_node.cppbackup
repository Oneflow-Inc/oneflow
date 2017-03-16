#include "graph/task_node.h"

namespace oneflow {

bool CompTaskNode::HasOpWithOutDiff() const {
  for (std::shared_ptr<const Operator> op : stage_node()->chain_node()->op_vec()) {
    if (! op->data_blob_name_set().output_diff_blob_names.empty()) {
      return true;
    }
  }
  return false;
}

bool CompTaskNode::HasOpWithIndiff() const {
  for (std::shared_ptr<const Operator> op : stage_node()->chain_node()->op_vec()) {
    if (! op->data_blob_name_set().input_diff_blob_names.empty()) {
      return true;
    }
  }
  return false;
}

} // namespace oneflow
