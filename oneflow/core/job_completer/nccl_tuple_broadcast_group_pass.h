#ifndef ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_BROADCAST_GROUP_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_BROADCAST_GROUP_PASS_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class OpGraph;
class Job;

class NcclTupleBroadcastGroupPass final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleBroadcastGroupPass);
  NcclTupleBroadcastGroupPass() = default;
  ~NcclTupleBroadcastGroupPass() = default;

  void Apply(const OpGraph& op_graph, Job* job);
};

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_BROADCAST_GROUP_PASS_H_
