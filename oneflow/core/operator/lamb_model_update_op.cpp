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
#include "oneflow/core/operator/lamb_model_update_op.h"

namespace oneflow {

void LAMBModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  const auto& lamb_conf = op_conf().lamb_model_update_conf().user_conf().lamb_conf();
  CHECK_GE(lamb_conf.beta1(), 0);
  CHECK_LT(lamb_conf.beta1(), 1);
  CHECK_GE(lamb_conf.beta2(), 0);
  CHECK_LT(lamb_conf.beta2(), 1);

  EnrollInputBn("m", false)->set_is_mutable(true);
  EnrollInputBn("v", false)->set_is_mutable(true);
  EnrollInputBn("beta1_t", false)->set_is_mutable(true);
  EnrollInputBn("beta2_t", false)->set_is_mutable(true);
  EnrollTmpBn("fw_buf");
}

Maybe<void> LAMBModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ_OR_RETURN(model_blob_desc->data_type(), job_desc().DefaultDataType());
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("m") == *model_blob_desc);
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("v") == *model_blob_desc);

  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("beta1_t")->shape(), Shape({1}));
  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("beta2_t")->shape(), Shape({1}));
  GetBlobDesc4BnInOp("beta1_t")->set_data_type(DataType::kFloat);
  GetBlobDesc4BnInOp("beta2_t")->set_data_type(DataType::kFloat);
  *GetBlobDesc4BnInOp("fw_buf") = *model_blob_desc;
  GetBlobDesc4BnInOp("fw_buf")->mut_shape() = Shape({2});

  return Maybe<void>::Ok();
}

const HashSet<std::string> LAMBModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{"beta1_t", "beta2_t", "fw_buf"};
}

const PbMessage& LAMBModelUpdateOp::GetCustomizedConf() const {
  return op_conf().lamb_model_update_conf();
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kLambConf, NormalModelUpdtOp, LAMBModelUpdateOp);

REGISTER_OP(OperatorConf::kLambModelUpdateConf, LAMBModelUpdateOp);

}  // namespace oneflow
