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
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

inline void SplitImpl(cfg::SbpSignature* sbp_sign, const std::string& bn, int64_t axis) {
  (*sbp_sign->mutable_bn_in_op2sbp_parallel())[bn].mutable_split_parallel()->set_axis(axis);
}

inline void BroadcastImpl(cfg::SbpSignature* sbp_sign, const std::string& bn) {
  (*sbp_sign->mutable_bn_in_op2sbp_parallel())[bn].mutable_broadcast_parallel();
}

inline void PartialSumImpl(cfg::SbpSignature* sbp_sign, const std::string& bn) {
  (*sbp_sign->mutable_bn_in_op2sbp_parallel())[bn].mutable_partial_sum_parallel();
}

}  // namespace

namespace user_op {

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Split(const OpArg& op_arg, int64_t axis) {
  SplitImpl(&sbp_sig_tmp_, GenRepeatedBn(op_arg.name(), op_arg.index()), axis);
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Split(const std::vector<OpArg>& op_args,
                                                            int64_t axis) {
  for (const auto& op_arg : op_args) { Split(op_arg, axis); }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Split(
    const std::vector<std::pair<std::string, int32_t>>& args, int64_t axis) {
  for (const auto& pair : args) {
    SplitImpl(&sbp_sig_tmp_, GenRepeatedBn(pair.first, pair.second), axis);
  }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Broadcast(const OpArg& op_arg) {
  BroadcastImpl(&sbp_sig_tmp_, GenRepeatedBn(op_arg.name(), op_arg.index()));
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Broadcast(const std::vector<OpArg>& op_args) {
  for (const auto& op_arg : op_args) { Broadcast(op_arg); }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Broadcast(
    const std::vector<std::pair<std::string, int32_t>>& op_args) {
  for (const auto& pair : op_args) {
    BroadcastImpl(&sbp_sig_tmp_, GenRepeatedBn(pair.first, pair.second));
  }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::PartialSum(const OpArg& op_arg) {
  PartialSumImpl(&sbp_sig_tmp_, GenRepeatedBn(op_arg.name(), op_arg.index()));
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::PartialSum(
    const std::vector<OpArg>& op_args) {
  for (const auto& op_arg : op_args) { PartialSum(op_arg); }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::PartialSum(
    const std::vector<std::pair<std::string, int32_t>>& op_args) {
  for (const auto& pair : op_args) {
    PartialSumImpl(&sbp_sig_tmp_, GenRepeatedBn(pair.first, pair.second));
  }
  return *this;
}

Maybe<void> GetSbpFnUtil::DefaultBroadcastToBroadcast(SbpContext* ctx) { return Maybe<void>::Ok(); }

Maybe<void> GetSbpFnUtil::SplitForEachAxis(SbpContext* ctx) {
  const auto& inputs = ctx->inputs();
  CHECK_GE_OR_RETURN(inputs.size(), 1)
      << "At least one input for op GetSbpFnUtil::SplitForEachAxis";
  int64_t num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex(inputs.at(0).first, inputs.at(0).second)
          .shape()
          .NumAxes();
  for (const auto& pair : inputs) {
    CHECK_EQ(
        num_axes,
        ctx->LogicalTensorDesc4InputArgNameAndIndex(pair.first, pair.second).shape().NumAxes());
  }
  for (int64_t axis = 0; axis < num_axes; ++axis) {
    ctx->NewBuilder().Split(inputs, axis).Split(ctx->outputs(), axis).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferNdSbp4SrcOp(user_op::InferParallelDistributionFnContext* ctx,
                             const std::string& default_sbp_str) {
  const Shape& hierarchy = ctx->parallel_hierarchy();
  const auto& sbp_str_list = ctx->user_op_conf().attr<std::vector<std::string>>("nd_sbp");

  // src op may have tick inputs whose sbp should be broadcast
  for (const auto& input_arg : ctx->inputs()) {
    cfg::ParallelDistribution* input_nd_sbp =
        ctx->ParallelDistribution4ArgNameAndIndex(input_arg.first, input_arg.second);
    FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
      input_nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
    }
  }

  for (const auto& output_arg : ctx->outputs()) {
    cfg::ParallelDistribution* output_nd_sbp =
        ctx->ParallelDistribution4ArgNameAndIndex(output_arg.first, output_arg.second);
    if (sbp_str_list.size() == 0) {
      // the default sbp of constant's output should be broadcast
      FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
        CHECK_OR_RETURN(
            ParseSbpParallelFromString(default_sbp_str, output_nd_sbp->add_sbp_parallel()));
      }
    } else {
      CHECK_EQ_OR_RETURN(sbp_str_list.size(), hierarchy.NumAxes());
      for (const std::string& sbp_str : sbp_str_list) {
        cfg::SbpParallel* sbp = output_nd_sbp->add_sbp_parallel();
        CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, sbp));
        CHECK_OR_RETURN(sbp->has_split_parallel() || sbp->has_broadcast_parallel());
      }
    }
  }

  return Maybe<void>::Ok();
}

}  // namespace user_op

}  // namespace oneflow
