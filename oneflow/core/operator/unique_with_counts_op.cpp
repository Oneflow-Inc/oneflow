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
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/unique_op_util.h"

namespace oneflow {

class UniqueWithCountsOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniqueWithCountsOp);
  UniqueWithCountsOp() = default;
  ~UniqueWithCountsOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;
  Maybe<void> InferInternalBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
};

void UniqueWithCountsOp::InitFromOpConf() {
  CHECK(op_conf().has_unique_with_counts_conf());
  EnrollInputBn("x", false);
  EnrollOutputBn("y", false);
  EnrollOutputBn("idx", false);
  EnrollOutputBn("count", false);
  EnrollOutputBn("num_unique", false);
  EnrollTmpBn("workspace");
}

namespace {

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  const BlobDesc* x = BlobDesc4BnInOp("x");
  CHECK_EQ_OR_RETURN(x->shape().NumAxes(), 1);
  BlobDesc* y = BlobDesc4BnInOp("y");
  *y = *x;
  const DataType idx_data_type = op_conf.unique_with_counts_conf().out_idx();
  CHECK(IsIndexDataType(idx_data_type));
  BlobDesc* idx = BlobDesc4BnInOp("idx");
  *idx = *x;
  idx->set_data_type(idx_data_type);
  BlobDesc* count = BlobDesc4BnInOp("count");
  *count = *x;
  count->set_data_type(idx_data_type);
  BlobDesc* num_unique = BlobDesc4BnInOp("num_unique");
  num_unique->mut_shape() = Shape({1});
  num_unique->set_data_type(idx_data_type);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> UniqueWithCountsOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
}

Maybe<void> UniqueWithCountsOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
}

Maybe<void> UniqueWithCountsOp::InferInternalBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  CHECK_EQ_OR_RETURN(x->shape().NumAxes(), 1);
  const DataType idx_data_type = op_conf().unique_with_counts_conf().out_idx();
  CHECK(IsIndexDataType(idx_data_type));
  BlobDesc* workspace = GetBlobDesc4BnInOp("workspace");
  workspace->set_data_type(DataType::kChar);
  int64_t workspace_size_in_bytes;
  UniqueOpUtil::GetUniqueWithCountsWorkspaceSizeInBytes(device_type(), x->data_type(),
                                                        idx_data_type, x->shape().elem_cnt(),
                                                        &workspace_size_in_bytes);
  workspace->mut_shape() = Shape({workspace_size_in_bytes});
  return Maybe<void>::Ok();
}

void UniqueWithCountsOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_unique_with_counts_conf()->set_indices_data_type(
      op_conf().unique_with_counts_conf().out_idx());
}

REGISTER_OP(OperatorConf::kUniqueWithCountsConf, UniqueWithCountsOp);

}  // namespace oneflow
