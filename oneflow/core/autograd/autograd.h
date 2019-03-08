#ifndef ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_H_
#define ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {

JobConf1 AutoGrad(const JobDesc& job_desc);

void GenerateBackwardOpConfIf(const Operator&, std::vector<OperatorConf>*,
                              const std::function<LogicalBlobId*(const std::string&)>&);

struct GenerateBackwardOpConfWrapperStruct final {
  using Func = std::function<void(const Operator&, std::vector<OperatorConf>*,
                                  const std::function<LogicalBlobId*(const std::string&)>&)>;
  GenerateBackwardOpConfWrapperStruct(const Func& f) : func(f) {}
  Func func;
};

#define REGISTER_OP_GRAD(op_type_case, gen_grad_func)                       \
  REGISTER_CLASS_CREATOR(op_type_case, GenerateBackwardOpConfWrapperStruct, \
                         ([] { return new GenerateBackwardOpConfWrapperStruct(gen_grad_func); }))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_H_
