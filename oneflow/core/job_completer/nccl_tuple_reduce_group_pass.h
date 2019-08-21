#ifndef ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_REDUCE_GROUP_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_REDUCE_GROUP_PASS_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class OpGraph;
class JobBuilder;

class NcclTupleReduceGroupPass final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleReduceGroupPass);
  NcclTupleReduceGroupPass() = default;
  ~NcclTupleReduceGroupPass() = default;

  void Apply(const OpGraph& op_graph, JobBuilder* job_builder);
};

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_JOB_COMPLETER_NCCL_TUPLE_REDUCE_GROUP_PASS_H_
