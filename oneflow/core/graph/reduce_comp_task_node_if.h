#ifndef ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_

#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceCompTaskNodeIf {
 public:
  virtual ~ReduceCompTaskNodeIf() = default;
  virtual void EnableMemSharingInReduce(
      std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) = 0;
};

int64_t InferRegstSize(const RegstDesc& regst);
void BuildCtrlRegstBetweenReduceCopyNodes(const CompTaskNode* src_reduce,
                                          const CompTaskNode* dst_reduce, int64_t copy_node_num);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
