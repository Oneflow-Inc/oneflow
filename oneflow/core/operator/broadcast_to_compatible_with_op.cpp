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
    FOR_RANGE(int, i, 0, op_conf().broadcast_to_compatible_with_conf().compatible_size()) {
      InputBlobModifier* modifer = MutInputBlobModifier4Ibn(GenRepeatedBn("compatible", i));
      modifer->set_use_header_only(true);
    }
    EnrollOutputBn("y");
  }

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().broadcast_to_compatible_with_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    int64_t num_compatibles = op_conf().broadcast_to_compatible_with_conf().compatible_size();
    const BlobDesc* x_desc = GetBlobDesc4BnInOp("x");
    Shape broadcasted_shape(x_desc->shape());
    FOR_RANGE(int64_t, i, 0, num_compatibles) {
      const BlobDesc* compatible_i = GetBlobDesc4BnInOp(GenRepeatedBn("compatible", i));
      GetBroadcastShape(broadcasted_shape, compatible_i->shape(), &broadcasted_shape);
    }
    BlobDesc* y_desc = GetBlobDesc4BnInOp("y");
    y_desc->CopyMetaFrom(*x_desc);
    y_desc->mut_shape() = broadcasted_shape;
    return Maybe<void>::Ok();
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

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    const Shape& y_shape = JUST(LogicalBlobDesc4Ibn("y"))->shape();
    int64_t broadcast_num_axes = y_shape.NumAxes();
    HashMap<std::string, Shape> ibn2extend_shape;
    for (const std::string bn : input_bns()) {
      const Shape& input_shape = JUST(LogicalBlobDesc4Ibn(bn))->shape();
      CHECK_OR_RETURN(
          ibn2extend_shape
              .emplace(bn, CreateLeftExtendedShape(ShapeView(input_shape), broadcast_num_axes))
              .second);
    }

    FOR_RANGE(int64_t, i, 0, broadcast_num_axes) {
      if (y_shape.At(i) == 1) { continue; }
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
