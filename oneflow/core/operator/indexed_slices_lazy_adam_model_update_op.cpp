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
#include "oneflow/core/operator/indexed_slices_reduce_sum_op_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

class IndexedSlicesLazyAdamMdUpdateOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IndexedSlicesLazyAdamMdUpdateOp);
  IndexedSlicesLazyAdamMdUpdateOp() = default;
  ~IndexedSlicesLazyAdamMdUpdateOp() override = default;

 private:
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return Maybe<void>::Ok();
  }
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext*, const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
      std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const override;
};

void IndexedSlicesLazyAdamMdUpdateOp::InitFromOpConf() {
  const auto& conf = op_conf().indexed_slices_lazy_adam_model_update_conf();
  CHECK_GE(conf.beta1(), 0);
  CHECK_LT(conf.beta1(), 1);
  CHECK_GE(conf.beta2(), 0);
  CHECK_LT(conf.beta2(), 1);

  EnrollInputBn("m", false)->set_is_mutable(true);
  EnrollInputBn("v", false)->set_is_mutable(true);
  EnrollInputBn("model_diff_indices", false);
  EnrollInputBn("model_diff_values", false);
  EnrollInputBn("model", false)->set_is_mutable(true);
  EnrollInputBn("train_step", false);
  EnrollInputBn("learning_rate", false);

  EnrollTmpBn("num_unique_diff_indices");
  EnrollTmpBn("unique_diff_indices");
  EnrollTmpBn("unique_diff_values");
  EnrollTmpBn("unique_workspace");
  EnrollTmpBn("local_learning_rate");
}

Maybe<void> IndexedSlicesLazyAdamMdUpdateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("model_diff_indices");
  const BlobDesc* values = GetBlobDesc4BnInOp("model_diff_values");
  const int64_t num_indices_axes = indices->shape().NumAxes();
  CHECK_GT(values->shape().NumAxes(), num_indices_axes);
  FOR_RANGE(int64_t, i, 0, num_indices_axes) {
    CHECK_EQ(indices->shape().At(i), values->shape().At(i));
  }
  *GetBlobDesc4BnInOp("unique_diff_indices") = *indices;
  *GetBlobDesc4BnInOp("unique_diff_values") = *values;
  BlobDesc* num_unique_diff_indices = GetBlobDesc4BnInOp("num_unique_diff_indices");
  num_unique_diff_indices->set_data_type(DataType::kInt32);
  num_unique_diff_indices->mut_shape() = Shape({1});
  int64_t unique_workspace_size = 0;
  IndexedSlicesReduceSumOpUtil::GetReduceSumWorkspaceSizeInBytes(
      device_type(), values->data_type(), indices->data_type(), indices->shape().elem_cnt(),
      values->shape().Count(num_indices_axes), &unique_workspace_size);
  BlobDesc* unique_workspace = GetBlobDesc4BnInOp("unique_workspace");
  unique_workspace->set_data_type(DataType::kChar);
  unique_workspace->mut_shape() = Shape({unique_workspace_size});
  BlobDesc* local_learning_rate = GetBlobDesc4BnInOp("local_learning_rate");
  local_learning_rate->mut_shape() = Shape({1});
  local_learning_rate->set_data_type(DataType::kFloat);
  return Maybe<void>::Ok();
}

Maybe<void> IndexedSlicesLazyAdamMdUpdateOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t num_indices_axes =
      JUST(LogicalBlobDesc4Ibn("model_diff_indices"))->shape().NumAxes();
  const int64_t num_model_axes = JUST(LogicalBlobDesc4Ibn("model"))->shape().NumAxes();
  SbpSignatureBuilder()
      .Broadcast("learning_rate")
      .Broadcast("train_step")
      .Broadcast("model_diff_indices")
      .Broadcast("model_diff_values")
      .Split("model", 0)
      .Split("m", 0)
      .Split("v", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  FOR_RANGE(int64_t, i, 1, num_model_axes) {
    SbpSignatureBuilder()
        .Broadcast("learning_rate")
        .Broadcast("train_step")
        .Broadcast("model_diff_indices")
        .Split("model_diff_values", num_indices_axes + i - 1)
        .Split("model", i)
        .Split("m", i)
        .Split("v", i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

const PbMessage& IndexedSlicesLazyAdamMdUpdateOp::GetCustomizedConf() const {
  return op_conf().indexed_slices_lazy_adam_model_update_conf();
}

void IndexedSlicesLazyAdamMdUpdateOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const {
  const BlobDesc& model_logical_blob_desc = LogicalBlobDesc4BnInOp("model");
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  const BlobDesc* indices_blob = GetBlobDesc4BnInOp("model_diff_indices");
  kernel_conf->set_data_type(model_blob_desc->data_type());
  const int64_t num_model_instances = model_logical_blob_desc.shape().At(0);
  IndexedSlicesLazyAdamModelUpdateKernelConf* indexed_slices_lazy_adam_model_update_conf =
      kernel_conf->mutable_indexed_slices_lazy_adam_model_update_conf();
  indexed_slices_lazy_adam_model_update_conf->set_indices_data_type(indices_blob->data_type());
  if (model_blob_desc->shape().At(0) == num_model_instances) {
    indexed_slices_lazy_adam_model_update_conf->set_lower_bound(0);
    indexed_slices_lazy_adam_model_update_conf->set_upper_bound(num_model_instances);
  } else {
    BalancedSplitter bs(num_model_instances, parallel_ctx->parallel_num());
    indexed_slices_lazy_adam_model_update_conf->set_lower_bound(
        bs.At(parallel_ctx->parallel_id()).begin());
    indexed_slices_lazy_adam_model_update_conf->set_upper_bound(
        bs.At(parallel_ctx->parallel_id()).end());
  }
}

REGISTER_OP(OperatorConf::kIndexedSlicesLazyAdamModelUpdateConf, IndexedSlicesLazyAdamMdUpdateOp);

}  // namespace oneflow
