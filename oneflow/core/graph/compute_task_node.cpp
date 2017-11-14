#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

void SortByParallelId(std::vector<CompTaskNode*>* node_vec) {
  std::sort(node_vec->begin(), node_vec->end(),
            [](const CompTaskNode* lhs, const CompTaskNode* rhs) {
              return lhs->parallel_ctx().parallel_id()
                     < rhs->parallel_ctx().parallel_id();
            });
}

}  // namespace oneflow
