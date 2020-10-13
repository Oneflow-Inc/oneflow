/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ParallelCastOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ParallelCastOp);
  ParallelCastOp() = default;
  ~ParallelCastOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
};

void ParallelCastOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_const_inplace_ibn("in");
}

Maybe<void> ParallelCastOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

Maybe<void> ParallelCastOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return NaiveInferBatchAxis(BatchAxis4BnInOp);
}

Maybe<void> ParallelCastOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  const ParallelCastOpConf& conf = op_conf().parallel_cast_conf();
  auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
  if (conf.has_split_axis()) {
    SbpParallel sbp_parallel;
    if (conf.split_axis().has_value()) {
      sbp_parallel.mutable_split_parallel()->set_axis(conf.split_axis().value());
    } else {
      sbp_parallel.mutable_broadcast_parallel();
    }
    (*bn2sbp)["in"] = sbp_parallel;
    (*bn2sbp)["out"] = sbp_parallel;
  } else {
    const SbpParallel& in_sbp_parallel = JUST(SbpInferHint4Ibn("in"))->sbp_parallel();
    (*bn2sbp)["in"] = in_sbp_parallel;
    (*bn2sbp)["out"] = in_sbp_parallel;
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kParallelCastConf, ParallelCastOp);

}  // namespace oneflow
