#ifndef ONEFLOW_CORE_JOB_COMPLETER_SPARSE_SOFTMAX_CROSS_ENTROPY_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_SPARSE_SOFTMAX_CROSS_ENTROPY_PASS_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class OpGraph;

class SparseSoftmaxCrossEntropyPass final {
 public:
  SparseSoftmaxCrossEntropyPass() = default;
  ~SparseSoftmaxCrossEntropyPass() = default;
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_SPARSE_SOFTMAX_CROSS_ENTROPY_PASS_H_
