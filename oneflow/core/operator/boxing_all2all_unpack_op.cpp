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

class BoxingAll2AllUnpackOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingAll2AllUnpackOp);
  BoxingAll2AllUnpackOp() = default;
  ~BoxingAll2AllUnpackOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 protected:
  virtual void VirtualInferBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}
  virtual void VirtualInitFromOpConf(){};

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

void BoxingAll2AllUnpackOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalBlobId BoxingAll2AllUnpackOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().boxing_all2all_unpack_conf().lbi();
}

LogicalBlobId BoxingAll2AllUnpackOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().boxing_all2all_unpack_conf().lbi();
}
const PbMessage& BoxingAll2AllUnpackOp::GetCustomizedConf() const {
  return op_conf().boxing_all2all_unpack_conf();
}

Maybe<void> BoxingAll2AllUnpackOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BoxingAll2AllUnpackOpConf& boxing_all2all_unpack_conf =
      this->op_conf().boxing_all2all_unpack_conf();
  const int64_t dst_split_axis = boxing_all2all_unpack_conf.dst_split_axis();
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *GetBlobDesc4BnInOp("in");

  Shape out_shape(boxing_all2all_unpack_conf.logical_shape());
  out_shape.Set(dst_split_axis,
                out_shape.At(dst_split_axis) / boxing_all2all_unpack_conf.parallel_num());
  out_blob_desc->mut_shape() = out_shape;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxingAll2AllUnpackConf, BoxingAll2AllUnpackOp);

}  // namespace oneflow
