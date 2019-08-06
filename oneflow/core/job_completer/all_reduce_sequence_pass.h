#ifndef ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_SEQUENCE_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_SEQUENCE_PASS_H_

#include "oneflow/core/job/job.pb.h"

namespace oneflow {

class OpGraph;

class AllReduceSequencePass final {
 public:
  AllReduceSequencePass() = default;
  ~AllReduceSequencePass() = default;
  void Apply(const OpGraph& op_graph, Job* job) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_SEQUENCE_PASS_H_
