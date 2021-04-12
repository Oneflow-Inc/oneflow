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
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> GenBroadcastToCompatibleWithGradOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_broadcast_to_compatible_with_conf());
  if (DiffLbi4BnInOp("x") != nullptr) {
    const Shape& x_shape = LogicalBlobDesc4BnInOp("x").shape();
    const Shape& y_shape = LogicalBlobDesc4BnInOp("y").shape();
    Shape x_extend_shape = CreateLeftExtendedShape(ShapeView(x_shape), y_shape.NumAxes());
    std::vector<int32_t> reduced_axes(x_extend_shape.NumAxes() - x_shape.NumAxes());
    std::iota(reduced_axes.begin(), reduced_axes.end(), 0);
    FOR_RANGE(int64_t, i, reduced_axes.size(), y_shape.NumAxes()) {
      if (x_extend_shape.At(i) == 1 && y_shape.At(i) != 1) {
        reduced_axes.push_back(i);
      } else {
        CHECK_EQ(x_extend_shape.At(i), y_shape.At(i));
      }
    }
    const auto reduce_sum_like_op =
        user_op::UserOpConfWrapperBuilder("System-AutoGrad-" + op.op_name())
            .Op("reduce_sum_like")
            .Input("x", GenLogicalBlobName(*DiffLbi4BnInOp("y")))
            .Input("like", GenLogicalBlobName(op.BnInOp2Lbi("x")))
            .Attr<std::vector<int32_t>>("axis", reduced_axes)
            .Output("y")
            .ScopeSymbolId(op.op_conf().scope_symbol_id())
            .Build();
    op_confs->push_back(reduce_sum_like_op.op_conf());
    *DiffLbi4BnInOp("x") = GenLogicalBlobId(reduce_sum_like_op.output("y", 0));
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBroadcastToCompatibleWithConf,
                 &GenBroadcastToCompatibleWithGradOpConf);

}  // namespace oneflow
