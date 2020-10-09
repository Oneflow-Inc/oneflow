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
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    const OptInt64* x_batch_axis = BatchAxis4BnInOp("x");
    *BatchAxis4BnInOp("y") = *x_batch_axis;
    *BatchAxis4BnInOp("idx") = *x_batch_axis;
    *BatchAxis4BnInOp("count") = *x_batch_axis;
    BatchAxis4BnInOp("num_unique")->clear_value();
    return Maybe<void>::Ok();
  }
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

Maybe<void> UniqueWithCountsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  CHECK_EQ_OR_RETURN(x->shape().NumAxes(), 1);
  BlobDesc* y = GetBlobDesc4BnInOp("y");
  *y = *x;
  const DataType idx_data_type = op_conf().unique_with_counts_conf().out_idx();
  CHECK(IsIndexDataType(idx_data_type));
  BlobDesc* idx = GetBlobDesc4BnInOp("idx");
  *idx = *x;
  idx->set_data_type(idx_data_type);
  BlobDesc* count = GetBlobDesc4BnInOp("count");
  *count = *x;
  count->set_data_type(idx_data_type);
  BlobDesc* num_unique = GetBlobDesc4BnInOp("num_unique");
  num_unique->mut_shape() = Shape({1});
  num_unique->set_data_type(idx_data_type);
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
