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
#include "oneflow/core/operator/lars_model_update_op.h"

namespace oneflow {

void LARSModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  EnrollInputBn("momentum", false)->set_is_mutable(true);
  EnrollTmpBn("lars_data_tmp");
}

Maybe<void> LARSModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("momentum") == *model_blob_desc);

  // lars_data_tmp for gpu compute
  // lars_data_tmp[0] for model_norm, lars_data_tmp[1] for model_diff_norm, lars_data_tmp[2] for
  // local_learning_rate
  *GetBlobDesc4BnInOp("lars_data_tmp") = *model_blob_desc;
  GetBlobDesc4BnInOp("lars_data_tmp")->mut_shape() = Shape({3});
  return Maybe<void>::Ok();
}

const HashSet<std::string> LARSModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{"lars_data_tmp"};
}

REGISTER_CLASS(int32_t, NormalModelUpdateOpUserConf::kLarsConf, NormalModelUpdtOp,
               LARSModelUpdateOp);

REGISTER_OP(OperatorConf::kLarsModelUpdateConf, LARSModelUpdateOp);

}  // namespace oneflow
