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
#include "oneflow/core/operator/indexed_slices_reduce_sum_op_util.h"

namespace oneflow {

class IndexedSlicesReduceSumOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IndexedSlicesReduceSumOp);
  IndexedSlicesReduceSumOp() = default;
  ~IndexedSlicesReduceSumOp() override = default;

  void InitFromOpConf() override;
  virtual Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const;
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

void IndexedSlicesReduceSumOp::InitFromOpConf() {
  CHECK(op_conf().has_indexed_slices_reduce_sum_conf());
  EnrollInputBn("x_indices", false);
  EnrollInputBn("x_values");
  EnrollOutputBn("y_indices", false);
  EnrollOutputBn("y_values");
  EnrollOutputBn("num_unique", false);
  EnrollTmpBn("workspace");
}

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  const BlobDesc* x_indices = BlobDesc4BnInOp("x_indices");
  const BlobDesc* x_values = BlobDesc4BnInOp("x_values");
  CHECK_LT_OR_RETURN(x_indices->shape().NumAxes(), x_values->shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, x_indices->shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(x_indices->shape().At(i), x_values->shape().At(i));
  }
  CHECK_OR_RETURN(IsIndexDataType(x_indices->data_type()));
  const int64_t n = x_indices->shape().elem_cnt();
  const int64_t m = x_values->shape().elem_cnt() / n;
  BlobDesc* y_indices = BlobDesc4BnInOp("y_indices");
  BlobDesc* y_values = BlobDesc4BnInOp("y_values");
  *y_indices = *x_indices;
  y_indices->mut_shape() = Shape({n});
  *y_values = *x_values;
  y_values->mut_shape() = Shape({n, m});
  BlobDesc* num_unique = BlobDesc4BnInOp("num_unique");
  num_unique->mut_shape() = Shape({1});
  num_unique->set_data_type(DataType::kInt64);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> IndexedSlicesReduceSumOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> IndexedSlicesReduceSumOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> IndexedSlicesReduceSumOp::InferInternalBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const BlobDesc* x_indices = GetBlobDesc4BnInOp("x_indices");
  const BlobDesc* x_values = GetBlobDesc4BnInOp("x_values");
  CHECK_LT_OR_RETURN(x_indices->shape().NumAxes(), x_values->shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, x_indices->shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(x_indices->shape().At(i), x_values->shape().At(i));
  }
  CHECK_OR_RETURN(IsIndexDataType(x_indices->data_type()));
  const int64_t n = x_indices->shape().elem_cnt();
  const int64_t m = x_values->shape().elem_cnt() / n;
  BlobDesc* workspace = GetBlobDesc4BnInOp("workspace");
  workspace->set_data_type(DataType::kChar);
  int64_t workspace_size_in_bytes;
  IndexedSlicesReduceSumOpUtil::GetReduceSumWorkspaceSizeInBytes(
      device_type(), x_values->data_type(), x_indices->data_type(), n, m, &workspace_size_in_bytes);
  workspace->mut_shape() = Shape({workspace_size_in_bytes});
  return Maybe<void>::Ok();
}

void IndexedSlicesReduceSumOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("x_values")->data_type());
  kernel_conf->mutable_indexed_slices_reduce_sum_conf()->set_indices_data_type(
      GetBlobDesc4BnInOp("x_indices")->data_type());
}

REGISTER_OP(OperatorConf::kIndexedSlicesReduceSumConf, IndexedSlicesReduceSumOp);

}  // namespace oneflow
