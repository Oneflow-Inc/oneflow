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

namespace oneflow {

class DynamicReshapeOp final : public Operator {
 public:
  void InitFromOpConf() {
    CHECK(op_conf().has_dynamic_reshape_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
  }
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    const DynamicReshapeOpConf& conf = op_conf().dynamic_reshape_conf();
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    DimVector out_dim_vec(conf.shape().dim().begin(), conf.shape().dim().end());
    if (parallel_ctx->parallel_num() > 1) {
      // consistent strategy
      //   ONLY support sbp: S(0); and -1 must at axis 0
      const auto& out_sbp_it = sbp_signature->bn_in_op2sbp_parallel().find("out");
      CHECK_OR_RETURN(out_sbp_it != sbp_signature->bn_in_op2sbp_parallel().end());
      const SbpParallel& out_sbp = out_sbp_it->second;
      const auto& in_sbp_it = sbp_signature->bn_in_op2sbp_parallel().find("in");
      CHECK_OR_RETURN(in_sbp_it != sbp_signature->bn_in_op2sbp_parallel().end());
      const SbpParallel& in_sbp = in_sbp_it->second;
      if (out_sbp.has_split_parallel()) {
        CHECK_EQ_OR_RETURN(out_sbp.split_parallel().axis(), 0);
        CHECK_EQ_OR_RETURN(out_dim_vec.at(0), -1);
        CHECK_OR_RETURN(in_sbp.has_split_parallel());
        CHECK_EQ_OR_RETURN(in_sbp.split_parallel().axis(), 0);
      }
    }
    int32_t inferred_axis = -1;
    int32_t product = 1;
    for (int32_t i = 0; i < out_dim_vec.size(); ++i) {
      if (out_dim_vec.at(i) == -1) {
        CHECK_EQ_OR_RETURN(-1, inferred_axis);
        inferred_axis = i;
      } else {
        CHECK_GT_OR_RETURN(out_dim_vec.at(i), 0);
        product *= out_dim_vec.at(i);
      }
    }
    if (inferred_axis >= 0) {
      CHECK_GE_OR_RETURN(product, 1);
      CHECK_EQ_OR_RETURN(in->shape().elem_cnt() % product, 0);
      out_dim_vec.at(inferred_axis) = in->shape().elem_cnt() / product;
    }
    out->mut_shape() = Shape(out_dim_vec);
    CHECK_EQ_OR_RETURN(in->shape().elem_cnt(), out->shape().elem_cnt());
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kDynamicReshapeConf, DynamicReshapeOp);

class DynamicReshapeLikeOp final : public Operator {
 public:
  void InitFromOpConf() {
    CHECK(op_conf().has_dynamic_reshape_like_conf());
    EnrollInputBn("x");
    EnrollOutputBn("y");
    EnrollInputBn("like", false);
  }
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("x")->shape().elem_cnt(),
                       GetBlobDesc4BnInOp("like")->shape().elem_cnt());
    GetBlobDesc4BnInOp("y")->CopyFrom(*GetBlobDesc4BnInOp("like"));
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kDynamicReshapeLikeConf, DynamicReshapeLikeOp);

}  // namespace oneflow
