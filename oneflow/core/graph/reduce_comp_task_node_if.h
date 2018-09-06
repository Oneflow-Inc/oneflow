#ifndef ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_

#include "oneflow/core/register/register_desc.h"

namespace oneflow {

class ReduceCompTaskNodeIf {
 public:
  virtual void EnableMemSharingInReduce(
      std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) = 0;
  virtual ~ReduceCompTaskNodeIf() = default;
};

int64_t InferRegstSize(const RegstDesc& regst);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
