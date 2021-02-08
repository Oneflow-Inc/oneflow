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
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

class BoxingIdentityOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingIdentityOp);
  BoxingIdentityOp() = default;
  ~BoxingIdentityOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

void BoxingIdentityOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalBlobId BoxingIdentityOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().boxing_identity_conf().lbi();
}

LogicalBlobId BoxingIdentityOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().boxing_identity_conf().lbi();
}

Maybe<void> BoxingIdentityOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxingIdentityConf, BoxingIdentityOp);

}  // namespace oneflow
