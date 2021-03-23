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
#include "oneflow/core/operator/scalar_op_base.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ScalarOpBase::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollInputBn("scalar");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
  ;
}

Maybe<void> ScalarOpBase::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* scalar_blob_desc = GetBlobDesc4BnInOp("scalar");
  CHECK_EQ_OR_RETURN(in_blob_desc->data_type(), scalar_blob_desc->data_type());
  CHECK_EQ_OR_RETURN(scalar_blob_desc->shape().elem_cnt(), 1);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> ScalarOpBase::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const Shape& in_shape = JUST(LogicalBlobDesc4Ibn("in")).shape();
  FOR_RANGE(int64_t, i, 0, in_shape.NumAxes()) {
    SbpSignatureBuilder().Split("in", i).Broadcast("scalar").Split("out", i).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  }
  JUST(VirtualGetSbpSignatures(LogicalBlobDesc4Ibn, sbp_sig_list));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
