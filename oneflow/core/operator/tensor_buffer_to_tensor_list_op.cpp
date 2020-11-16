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
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class TensorBufferToTensorListOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorBufferToTensorListOp);
  TensorBufferToTensorListOp() = default;
  ~TensorBufferToTensorListOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_tensor_buffer_to_tensor_list_conf());
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false)->set_header_infered_before_compute(false);
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* in_desc = GetBlobDesc4BnInOp("in");
    CHECK_EQ_OR_RETURN(in_desc->data_type(), DataType::kTensorBuffer);
    CHECK_EQ_OR_RETURN(in_desc->shape().NumAxes(), 1);
    DimVector dim_vec = in_desc->shape().dim_vec();
    const ShapeProto& shape = op_conf().tensor_buffer_to_tensor_list_conf().shape();
    dim_vec.insert(dim_vec.end(), shape.dim().begin(), shape.dim().end());
    BlobDesc* out_desc = GetBlobDesc4BnInOp("out");
    out_desc->mut_shape() = Shape(dim_vec);
    out_desc->set_data_type(op_conf().tensor_buffer_to_tensor_list_conf().data_type());
    out_desc->set_is_tensor_list(true);
    out_desc->set_is_dynamic(true);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    CHECK_OR_RETURN(BatchAxis4BnInOp("in")->has_value());
    CHECK_EQ_OR_RETURN(BatchAxis4BnInOp("in")->value(), 0);
    BatchAxis4BnInOp("out")->set_value(0);
    return Maybe<void>::Ok();
  }
};

REGISTER_CPU_OP(OperatorConf::kTensorBufferToTensorListConf, TensorBufferToTensorListOp);

}  // namespace oneflow
