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
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

namespace {

Maybe<void> GetBroadcastShape(const Shape& a_shape, const Shape& b_shape, Shape* broadcast_shape) {
  Shape max_shape = Shape::Ones(std::max(a_shape.NumAxes(), b_shape.NumAxes()));
  Shape a_extend_shape = CreateLeftExtendedShape(ShapeView(a_shape), max_shape.NumAxes());
  Shape b_extend_shape = CreateLeftExtendedShape(ShapeView(b_shape), max_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, max_shape.NumAxes()) {
    CHECK_OR_RETURN(a_extend_shape.At(i) == 1 || b_extend_shape.At(i) == 1
                    || a_extend_shape.At(i) == b_extend_shape.At(i))
        << "shape " << a_shape.ToString() << " and shape " << b_shape.ToString()
        << " are not broadcastable";
    max_shape.Set(i, std::max(a_extend_shape.At(i), b_extend_shape.At(i)));
  }
  *broadcast_shape = max_shape;
  return Maybe<void>::Ok();
}

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  int64_t num_compatibles = op_conf.broadcast_to_compatible_with_conf().compatible_size();
  const BlobDesc* x_desc = BlobDesc4BnInOp("x");
  Shape broadcasted_shape(x_desc->shape());
  FOR_RANGE(int64_t, i, 0, num_compatibles) {
    const BlobDesc* compatible_i = BlobDesc4BnInOp(GenRepeatedBn("compatible", i));
    GetBroadcastShape(broadcasted_shape, compatible_i->shape(), &broadcasted_shape);
  }
  BlobDesc* y_desc = BlobDesc4BnInOp("y");
  y_desc->CopyFrom(*x_desc);
  y_desc->mut_shape() = broadcasted_shape;
  return Maybe<void>::Ok();
}

}  // namespace

class BroadcastToCompatibleWithOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastToCompatibleWithOp);
  BroadcastToCompatibleWithOp() = default;
  ~BroadcastToCompatibleWithOp() override = default;

  void InitFromOpConf() {
    CHECK(op_conf().has_broadcast_to_compatible_with_conf());
    EnrollInputBn("x");
    EnrollRepeatedInputBn("compatible", false);
    EnrollOutputBn("y");
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
  }

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
    auto* conf = kernel_conf->mutable_broadcast_to_compatible_with_conf();
    const BlobDesc* x_desc = GetBlobDesc4BnInOp("x");
    const BlobDesc* y_desc = GetBlobDesc4BnInOp("y");
    Shape x_extend_shape =
        CreateLeftExtendedShape(ShapeView(x_desc->shape()), y_desc->shape().NumAxes());
    FOR_RANGE(int64_t, i, 0, y_desc->shape().NumAxes()) {
      if (x_extend_shape.At(i) == 1 && y_desc->shape().At(i) != 1)
        conf->mutable_broadcast_axes()->Add(i);
    }
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    Shape broadcasted_shape{1};
    for (const std::string ibn : input_bns()) {
      const Shape& input_shape = JUST(LogicalBlobDesc4Ibn(ibn)).shape();
      GetBroadcastShape(broadcasted_shape, input_shape, &broadcasted_shape);
    }

    const int64_t broadcast_num_axes = broadcasted_shape.NumAxes();
    HashMap<std::string, Shape> ibn2extend_shape;
    for (const std::string ibn : input_bns()) {
      const Shape& input_shape = JUST(LogicalBlobDesc4Ibn(ibn)).shape();
      CHECK_OR_RETURN(
          ibn2extend_shape
              .emplace(ibn, CreateLeftExtendedShape(ShapeView(input_shape), broadcast_num_axes))
              .second);
    }

    FOR_RANGE(int64_t, i, 0, broadcast_num_axes) {
      if (broadcasted_shape.At(i) == 1) { continue; }
      SbpSignature sbp_sig;
      for (const auto& pair : ibn2extend_shape) {
        if (pair.second.At(i) == 1) {
          (*sbp_sig.mutable_bn_in_op2sbp_parallel())[pair.first].mutable_broadcast_parallel();
        } else {
          (*sbp_sig.mutable_bn_in_op2sbp_parallel())[pair.first].mutable_split_parallel()->set_axis(
              i - (broadcast_num_axes - pair.second.NumAxes()));
        }
      }
      (*sbp_sig.mutable_bn_in_op2sbp_parallel())["y"].mutable_split_parallel()->set_axis(i);
      *sbp_sig_list->mutable_sbp_signature()->Add() = sbp_sig;
    }

    PbRpf<std::string> compatible_bns;
    int64_t num_compatibles = op_conf().broadcast_to_compatible_with_conf().compatible_size();
    FOR_RANGE(int64_t, i, 0, num_compatibles) {
      *compatible_bns.Add() = GenRepeatedBn("compatible", i);
    }
    SbpSignatureBuilder()
        .PartialSum("x")
        .Broadcast(compatible_bns)
        .PartialSum("y")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    SbpSignatureBuilder()
        .Broadcast("x")
        .PartialSum(compatible_bns)
        .Broadcast("y")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kBroadcastToCompatibleWithConf, BroadcastToCompatibleWithOp);

}  // namespace oneflow
