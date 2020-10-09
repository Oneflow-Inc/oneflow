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
#include "oneflow/core/operator/foreign_input_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

void CheckOpConf(const OperatorConf& op_conf) { CHECK(op_conf.ctrl_in_op_name().empty()); }

}  // namespace

void ForeignInputOp::InitFromOpConf() {
  CHECK(op_conf().has_foreign_input_conf());
  if (op_conf().foreign_input_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

Maybe<void> ForeignInputOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  CheckOpConf(op_conf());
  const auto& conf = op_conf().foreign_input_conf().blob_conf();
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(conf.shape());
  if (conf.has_data_type()) {
    out_blob_desc->set_data_type(conf.data_type());
  } else {
    out_blob_desc->set_data_type(job_desc().DefaultDataType());
  }
  out_blob_desc->set_is_dynamic(conf.is_dynamic());
  out_blob_desc->set_is_tensor_list(conf.is_tensor_list());
  return Maybe<void>::Ok();
}

Maybe<void> ForeignInputOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = op_conf().foreign_input_conf().blob_conf().batch_axis();
  return Maybe<void>::Ok();
}

Maybe<void> ForeignInputOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kForeignInputConf, ForeignInputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kForeignInputConf, 1);

}  // namespace oneflow
