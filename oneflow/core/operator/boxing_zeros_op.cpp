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

namespace oneflow {

class BoxingZerosOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingZerosOp);
  BoxingZerosOp() = default;
  ~BoxingZerosOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

void BoxingZerosOp::InitFromOpConf() { EnrollOutputBn("out", false); }

LogicalBlobId BoxingZerosOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().boxing_zeros_conf().lbi();
}

LogicalBlobId BoxingZerosOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().boxing_zeros_conf().lbi();
}

Maybe<void> BoxingZerosOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const BoxingZerosOpConf& conf = this->op_conf().boxing_zeros_conf();
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(conf.data_type());
  out->mut_shape() = Shape(conf.shape());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxingZerosConf, BoxingZerosOp);

}  // namespace oneflow
