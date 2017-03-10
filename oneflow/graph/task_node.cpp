#include "graph/task_node.h"

namespace oneflow {

bool ComputeTnd::HasOpWithOutDiff() const {
  for (std::shared_ptr<const Operator> op : op_vec_) {
    if (! op->data_blob_name_set().output_diff_blob_names.empty()) {
      return true;
    }
  }
  return false;
}

bool ComputeTnd::HasOpWithIndiff() const {
  for (std::shared_ptr<const Operator> op : op_vec_) {
    if (! op->data_blob_name_set().input_diff_blob_names.empty()) {
      return true;
    }
  }
  return false;
}

} // namespace oneflow
