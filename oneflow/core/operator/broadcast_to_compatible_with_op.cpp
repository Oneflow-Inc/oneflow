#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

Maybe<void> GetBroadcastShape(const Shape& a_shape, const Shape& b_shape, Shape* broadcast_shape) {
  Shape max_shape = Shape::Ones(std::max(a_shape.NumAxes(), b_shape.NumAxes()));
  Shape a_extend_shape = CreateLeftExtendedShape(ShapeView(a_shape), max_shape.NumAxes());
  Shape b_extend_shape = CreateLeftExtendedShape(ShapeView(b_shape), max_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, max_shape.NumAxes()) {
    OF_CHECK(a_extend_shape.At(i) == 1 || b_extend_shape.At(i) == 1
             || a_extend_shape.At(i) == b_extend_shape.At(i))
        << "shape " << a_shape.ToString() << " and shape " << b_shape.ToString()
        << " are not broadcastable";
    max_shape.Set(i, std::max(a_extend_shape.At(i), b_extend_shape.At(i)));
  }
  *broadcast_shape = max_shape;
  return Maybe<void>::Ok();
}

void GenBroadcastToCompatibleWithGradOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_broadcast_to_compatible_with_conf());
  if (DiffLbi4BnInOp("x") != nullptr) {
    const BlobDesc& x_desc = LogicalBlobDesc4BnInOp("x");
    const BlobDesc& y_desc = LogicalBlobDesc4BnInOp("y");
    Shape x_extend_shape = CreateLeftExtendedShape(x_desc.shape(), y_desc.shape().NumAxes());
    std::vector<int32_t> reduced_axes;
    FOR_RANGE(int64_t, i, 0, y_desc.shape().NumAxes()) {
      if (x_extend_shape.At(i) == 1) {
        reduced_axes.push_back(i);
      } else {
        CHECK_EQ(x_extend_shape.At(i), y_desc.shape().At(i));
      }
    }

    OperatorConf reduce_sum_like_op;
    reduce_sum_like_op.set_name("System-AutoGrad-" + op.op_name());
    ReduceSumLikeOpConf* conf = reduce_sum_like_op.mutable_reduce_sum_like_conf();
    conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("y")));
    conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("x")));
    conf->set_y("y");
    *conf->mutable_axis() = StdVec2PbRf(reduced_axes);
    op_confs->push_back(reduce_sum_like_op);
    DiffLbi4BnInOp("x")->set_op_name(reduce_sum_like_op.name());
    DiffLbi4BnInOp("x")->set_blob_name(conf->y());
  }
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
    EnrollOutputBn("y")->set_const_inplace_ibn("x");
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
    const Shape& x_shape = JUST(LogicalBlobDesc4Ibn("x"))->shape();
    const Shape& y_shape = JUST(LogicalBlobDesc4Ibn("y"))->shape();
    int64_t broadcast_num_axes = y_shape.NumAxes();
    int64_t num_compatibles = op_conf().broadcast_to_compatible_with_conf().compatible_size();
    std::vector<Shape> compatible_extend_shape_vec(num_compatibles);
    FOR_RANGE(int64_t, i, 0, num_compatibles) {
      compatible_extend_shape_vec.at(i) =
          JUST(LogicalBlobDesc4Ibn(GenRepeatedBn("compatible", i)))->shape();
    }
    std::transform(compatible_extend_shape_vec.begin(), compatible_extend_shape_vec.end(),
                   compatible_extend_shape_vec.begin(), [=](Shape shape) {
                     return CreateLeftExtendedShape(ShapeView(shape), broadcast_num_axes);
                   });
    Shape x_extend_shape = CreateLeftExtendedShape(ShapeView(x_shape), broadcast_num_axes);

    FOR_RANGE(int64_t, i, 0, broadcast_num_axes) {
      SbpSignature sbp_sig;
      int64_t split_axis = -1;
      if (x_extend_shape.At(i) == 1) {
        (*sbp_sig.mutable_bn_in_op2sbp_parallel())["x"].mutable_broadcast_parallel();
      } else {
        (*sbp_sig.mutable_bn_in_op2sbp_parallel())["x"].mutable_split_parallel()->set_axis(
            i - (broadcast_num_axes - x_shape.NumAxes()));
        split_axis = i;
      }
      FOR_RANGE(int64_t, j, 0, num_compatibles) {
        std::string compatible_bn = GenRepeatedBn("compatible", j);
        if (compatible_extend_shape_vec.at(j).At(i) == 1) {
          (*sbp_sig.mutable_bn_in_op2sbp_parallel())[compatible_bn].mutable_broadcast_parallel();
        } else {
          (*sbp_sig.mutable_bn_in_op2sbp_parallel())[compatible_bn]
              .mutable_split_parallel()
              ->set_axis(i - (broadcast_num_axes - compatible_extend_shape_vec.at(j).NumAxes()));
          split_axis = i;
        }
      }
      if (split_axis >= 0) {
        (*sbp_sig.mutable_bn_in_op2sbp_parallel())["y"].mutable_split_parallel()->set_axis(i);
        *sbp_sig_list->mutable_sbp_signature()->Add() = sbp_sig;
      }
    }

    PbRpf<std::string> compatible_bns;
    FOR_RANGE(int64_t, i, 0, num_compatibles) {
      *compatible_bns.Add() = GenRepeatedBn("compatible", i);
    }
    SbpSignatureBuilder()
        .PartialSum("x")
        .Broadcast(compatible_bns)
        .PartialSum("y")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kBroadcastToCompatibleWithConf, BroadcastToCompatibleWithOp);
REGISTER_OP_GRAD(OperatorConf::kBroadcastToCompatibleWithConf,
                 &GenBroadcastToCompatibleWithGradOpConf);

}  // namespace oneflow
