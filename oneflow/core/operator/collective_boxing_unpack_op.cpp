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

class CollectiveBoxingUnpackOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingUnpackOp);
  CollectiveBoxingUnpackOp() = default;
  ~CollectiveBoxingUnpackOp() override = default;

  void InitFromOpConf() override;

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    UNIMPLEMENTED_THEN_RETURN();
  }
  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

void CollectiveBoxingUnpackOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalBlobId CollectiveBoxingUnpackOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().collective_boxing_unpack_conf().lbi();
}

LogicalBlobId CollectiveBoxingUnpackOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().collective_boxing_unpack_conf().lbi();
}

Maybe<void> CollectiveBoxingUnpackOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const CollectiveBoxingUnpackOpConf& unpack_conf = this->op_conf().collective_boxing_unpack_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;

  Shape out_shape(unpack_conf.logical_shape());
  if (unpack_conf.dst_sbp_parallel().has_split_parallel()) {
    const int64_t dst_split_axis = unpack_conf.dst_sbp_parallel().split_parallel().axis();
    out_shape.Set(dst_split_axis, out_shape.At(dst_split_axis) / unpack_conf.num_ranks());
  }
  CHECK_EQ_OR_RETURN(out_shape.elem_cnt(), in_blob_desc->shape().elem_cnt());
  out_blob_desc->mut_shape() = out_shape;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kCollectiveBoxingUnpackConf, CollectiveBoxingUnpackOp);

}  // namespace oneflow
