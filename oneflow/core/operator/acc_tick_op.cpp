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
#include "oneflow/core/operator/acc_tick_op.h"

namespace oneflow {

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp) {
  *GetBlobDesc4BnInOp("acc") = *GetBlobDesc4BnInOp("one");
  GetBlobDesc4BnInOp("acc")->mut_shape() = Shape({1LL});
  return Maybe<void>::Ok();
}

}  // namespace

void AccTickOp::InitFromOpConf() {
  CHECK(op_conf().has_acc_tick_conf());

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

Maybe<void> AccTickOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> AccTickOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> AccTickOp::InferOpTimeShape(
    const std::function<Maybe<const Shape>(const std::string&)>& GetTimeShape4BnInOp,
    std::shared_ptr<const Shape>* time_shape) const {
  const int32_t max_acc_num = op_conf().acc_tick_conf().max_acc_num();
  std::shared_ptr<const Shape> in_shape = JUST(GetTimeShape4BnInOp("one"));
  CHECK_EQ_OR_RETURN(in_shape->elem_cnt() % max_acc_num, 0);
  DimVector in_dim_vec = in_shape->dim_vec();
  std::shared_ptr<Shape> op_time_shape;
  if (in_dim_vec.back() == max_acc_num) {
    in_dim_vec.pop_back();
    op_time_shape.reset(new Shape(in_dim_vec));
  } else if (in_dim_vec.back() % max_acc_num == 0) {
    in_dim_vec.back() /= max_acc_num;
    op_time_shape.reset(new Shape(in_dim_vec));
  } else {
    op_time_shape.reset(new Shape({in_shape->elem_cnt() / max_acc_num}));
  }
  *time_shape = op_time_shape;
  return Maybe<void>::Ok();
}

Maybe<void> AccTickOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kAccTickConf, AccTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kAccTickConf);

}  // namespace oneflow
