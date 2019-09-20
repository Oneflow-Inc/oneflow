#ifndef ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class OpGraph;

class AllReduceAddPass final {
 public:
  AllReduceAddPass() = default;
  ~AllReduceAddPass() = default;
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_
