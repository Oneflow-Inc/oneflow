#ifndef ONEFLOW_CORE_OPERATOR_PARALLEL_CAST_OP_H_
#define ONEFLOW_CORE_OPERATOR_PARALLEL_CAST_OP_H_

#include "oneflow/core/operator/identity_op.h"

namespace oneflow {

class ParallelCastOp final : public IdentityOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ParallelCastOp);
  ParallelCastOp() = default;
  ~ParallelCastOp() override = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().parallel_cast_conf(); }

 private:
  void InferSbpSignature(SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
                         const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
                         std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
                         const ParallelDesc& parallel_desc) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PARALLEL_CAST_OP_H_
