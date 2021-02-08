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
#include "oneflow/core/operator/cwise_op.h"

namespace oneflow {

void CWiseOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in_0");
  VirtualInitFromOpConf();
}

Maybe<void> CWiseOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const auto* blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    CHECK_OR_RETURN(*in_0_blob_desc == *blob_desc);
  }
  *GetBlobDesc4BnInOp("out") = *in_0_blob_desc;
  return VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

}  // namespace oneflow
