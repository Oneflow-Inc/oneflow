#ifndef ONEFLOW_MLIRONEFLOWTRANSLATION_H
#define ONEFLOW_MLIRONEFLOWTRANSLATION_H

#include "oneflow/core/job/job.pb.h"
#include <functional>
namespace mlir {

class RoundTripOneFlowJobWrapperInterface {
 public:
  virtual const ::oneflow::Job* job() const = 0;
  virtual const ::oneflow::ParallelConf& ParallelConf4OpName(const std::string& op_name) = 0;
};

void RoundTripOneFlowJob(
    const RoundTripOneFlowJobWrapperInterface& job_wrapper,
    std::function<bool(::oneflow::Job* job, std::string& reason)> is_legit_job);
void registerFromOneFlowJobTranslation();

}  // namespace mlir

#endif /* ONEFLOW_MLIRONEFLOWTRANSLATION_H */
