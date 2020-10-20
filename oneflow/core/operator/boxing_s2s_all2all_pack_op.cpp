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

class BoxingS2SAll2AllPackOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingS2SAll2AllPackOp);
  BoxingS2SAll2AllPackOp() = default;
  ~BoxingS2SAll2AllPackOp() override = default;

  void InitFromOpConf() override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 protected:
  virtual void VirtualInferBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

void BoxingS2SAll2AllPackOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalBlobId BoxingS2SAll2AllPackOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().boxing_s2s_all2all_pack_conf().lbi();
}

LogicalBlobId BoxingS2SAll2AllPackOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().boxing_s2s_all2all_pack_conf().lbi();
}

Maybe<void> BoxingS2SAll2AllPackOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().elem_cnt()});
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxingS2SAll2AllPackConf, BoxingS2SAll2AllPackOp);

}  // namespace oneflow
