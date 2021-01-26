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

class BoxingPackTransposeOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingPackTransposeOp);
  BoxingPackTransposeOp() = default;
  ~BoxingPackTransposeOp() override = default;

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

void BoxingPackTransposeOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalBlobId BoxingPackTransposeOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().boxing_pack_transpose_conf().lbi();
}

LogicalBlobId BoxingPackTransposeOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().boxing_pack_transpose_conf().lbi();
}

Maybe<void> BoxingPackTransposeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().elem_cnt()});
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxingPackTransposeConf, BoxingPackTransposeOp);

}  // namespace oneflow
