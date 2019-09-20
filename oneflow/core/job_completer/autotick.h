#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTOTICK_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTOTICK_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void AutoSourceTick(const OpGraph& op_graph, JobBuilder* job_builder);
void AddTickForTimeShape(const OpGraph& op_graph, JobBuilder* job_builder);
void AutoSinkTick(const OpGraph& op_graph, JobBuilder* job_builder);
void AddGlobalTotalJobCriticalSection(const Job& job);
void AddGlobalInputCriticalSections(const OpGraph& op_graph, JobBuilder* job_builder);
void AddGlobalOutputCriticalSections(const OpGraph& op_graph, JobBuilder* job_builder);

class MutOpConTickInputHelper {
 public:
  bool IsTickInputBound() const {
    return !op_conf_->ctrl_in_op_name().empty() || VirtualIsTickInputBound();
  }
  virtual bool VirtualIsTickInputBound() const = 0;
  virtual OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const = 0;
  void InitFromOpConf(const OperatorConf& op_conf) { op_conf_ = &op_conf; }

 protected:
  MutOpConTickInputHelper() : op_conf_(nullptr) {}
  const OperatorConf& op_conf() const { return *op_conf_; }

 private:
  const OperatorConf* op_conf_;
};

#define REGISTER_AUTO_TICK(op_type_case, HelperType) \
  REGISTER_CLASS(op_type_case, MutOpConTickInputHelper, HelperType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTOTICK_H_
