#ifndef ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_BROADCAST_REDUCE_SEQUENCE_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_BROADCAST_REDUCE_SEQUENCE_PASS_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class OpGraph;
class JobBuilder;

class NcclTupleBroadcastReduceSequencePass final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleBroadcastReduceSequencePass);
  NcclTupleBroadcastReduceSequencePass() = default;
  ~NcclTupleBroadcastReduceSequencePass() = default;

  void Apply(const OpGraph& op_graph, JobBuilder* job_builder);
};

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_BROADCAST_REDUCE_SEQUENCE_PASS_H_
