#ifndef ONEFLOW_MLIRONEFLOWTRANSLATION_H
#define ONEFLOW_MLIRONEFLOWTRANSLATION_H

#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include <functional>
#include <string>
namespace mlir {

class RoundTripOneFlowJobWrapperInterface {
 public:
  virtual const ::oneflow::Job* job() const = 0;
  virtual void UpdateJob(const ::oneflow::Job* new_job) = 0;
  virtual void DumpMLIR(const std::string& filename, const std::string& content) = 0;
  virtual const ::oneflow::ParallelConf& ParallelConf4OpName(const std::string& op_name) const = 0;
  virtual const ::oneflow::OperatorConf& OpConf4OpName(const std::string& op_name) const = 0;
  virtual std::pair<std::vector<std::string>, std::vector<std::string>> InputBns4OpName(
      const std::string& op_name) const = 0;
  virtual std::vector<std::string> OutputLbns4OpName(const std::string& op_name) const = 0;
  virtual std::string ReplaceInputLbnInOpCustomizedConf(::oneflow::OperatorConf* op_conf,
                                                        const std::string& ibn,
                                                        const std::string& new_val) const = 0;
  virtual ::oneflow::AttrType QueryAttrType(const std::string& op_type_name,
                                            const std::string& attr_name) const = 0;
  virtual void TopoForEachOpConf(
      std::function<void(const ::oneflow::OperatorConf*)> Handler) const = 0;
};

void RoundTripOneFlowJob(
    RoundTripOneFlowJobWrapperInterface& job_wrapper,
    std::function<bool(::oneflow::Job* job, std::string& reason)> is_legit_job);
void registerFromOneFlowJobTranslation();

}  // namespace mlir

#endif /* ONEFLOW_MLIRONEFLOWTRANSLATION_H */
