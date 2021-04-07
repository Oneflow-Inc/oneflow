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
#include "oneflow/core/operator/device_tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void DeviceTickOp::InitFromOpConf() {
  CHECK(op_conf().has_device_tick_conf());
  EnrollRepeatedInputBn("tick", false);
  EnrollOutputBn("out", false);
}

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  BlobDesc* blob_desc = BlobDesc4BnInOp("out");
  blob_desc->mut_shape() = Shape({1});
  blob_desc->set_data_type(DataType::kUInt8);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> DeviceTickOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> DeviceTickOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> DeviceTickOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

Maybe<void> DeviceTickOp::InferOpTimeShape(
    const std::function<Maybe<const Shape>(const std::string&)>& GetTimeShape4BnInOp,
    std::shared_ptr<const Shape>* time_shape) const {
  std::shared_ptr<const Shape> in_time_shape;
  for (const auto& bn : input_bns()) {
    std::shared_ptr<const Shape> ts = JUST(GetTimeShape4BnInOp(bn));
    if (!in_time_shape) {
      in_time_shape = ts;
    } else {
      CHECK_OR_RETURN(*in_time_shape == *ts);
    }
  }
  if (this->op_conf().device_tick_conf().has_time_shape()) {
    if (!in_time_shape) {
      in_time_shape.reset(new Shape(this->op_conf().device_tick_conf().time_shape()));
    } else {
      CHECK_OR_RETURN(in_time_shape->elem_cnt()
                      == Shape(this->op_conf().device_tick_conf().time_shape()).elem_cnt());
    }
  }
  if (in_time_shape) {
    *time_shape = in_time_shape;
  } else {
    *time_shape = std::make_shared<const Shape>(Shape({1, 1}));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kDeviceTickConf, 2);
REGISTER_OP(OperatorConf::kDeviceTickConf, DeviceTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kDeviceTickConf);

}  // namespace oneflow
