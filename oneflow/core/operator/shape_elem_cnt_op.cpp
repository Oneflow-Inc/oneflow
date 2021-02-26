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
#include "oneflow/core/operator/shape_elem_cnt_op.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

HashSet<int32_t> GetInclusiveAxes(const ShapeElemCntOpConf& conf, int32_t num_axes) {
  HashSet<int32_t> ret;
  if (conf.has_exclude_axis_conf()) {
    HashSet<int32_t> exclude_axes(conf.exclude_axis_conf().axis().begin(),
                                  conf.exclude_axis_conf().axis().end());
    FOR_RANGE(int32_t, i, 0, num_axes) {
      if (exclude_axes.find(i) == exclude_axes.end()
          && exclude_axes.find(i - num_axes) == exclude_axes.end()) {
        ret.insert(i);
      }
    }
  } else if (conf.has_include_axis_conf()) {
    for (int32_t axis : conf.include_axis_conf().axis()) {
      if (axis < 0) { axis += num_axes; }
      CHECK_GE(axis, 0);
      CHECK_LT(axis, num_axes);
      ret.insert(axis);
    }
  } else if (conf.has_range_axis_conf()) {
    TODO();
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

}  // namespace

void ShapeElemCntOp::InitFromOpConf() {
  EnrollInputBn("x", false);
  EnrollOutputBn("y", false);
}

namespace {

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  BlobDesc4BnInOp("y")->set_data_type(op_conf.shape_elem_cnt_conf().data_type());
  BlobDesc4BnInOp("y")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> ShapeElemCntOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
}

Maybe<void> ShapeElemCntOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
}

void ShapeElemCntOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  int32_t num_axes = GetBlobDesc4BnInOp("x")->shape().NumAxes();
  const HashSet<int32_t>& inclusive_axis =
      GetInclusiveAxes(op_conf().shape_elem_cnt_conf(), num_axes);
  *kernel_conf->mutable_shape_elem_cnt_conf()->mutable_axis() = {inclusive_axis.begin(),
                                                                 inclusive_axis.end()};
}

Maybe<void> ShapeElemCntOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int32_t num_axes = JUST(LogicalBlobDesc4Ibn("x")).shape().NumAxes();
  const auto& inclusive_axes = GetInclusiveAxes(op_conf().shape_elem_cnt_conf(), num_axes);
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(inclusive_axes, num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .PartialSum(output_bns())
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    } else {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .Broadcast(output_bns())
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kShapeElemCntConf, ShapeElemCntOp);

}  // namespace oneflow
