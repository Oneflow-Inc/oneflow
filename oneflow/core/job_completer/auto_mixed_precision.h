#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_

#include "oneflow/core/job_completer/auto_mixed_precision_lists.h"

namespace oneflow {

class OpGraph;
class Job;

class AutoMixedPrecision final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoMixedPrecision);
  AutoMixedPrecision(const AMPList& white, const AMPList& black, const AMPList& gray)
      : white_list_(white), black_list_(black), gray_list_(gray) {}
  ~AutoMixedPrecision() = default;

  void Apply(const OpGraph& op_graph, Job* job);

 private:
  const AMPList& white_list_;
  const AMPList& black_list_;
  const AMPList& gray_list_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_
