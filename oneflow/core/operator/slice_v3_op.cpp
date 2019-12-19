#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

int64_t FixSliceBegin(int64_t begin, int64_t dims) {
  begin = (begin >= 0) ? begin : begin + dims;
  CHECK_GE(begin, 0);
  CHECK_LT(begin, dims);
  return begin;
}

int64_t FixSliceEnd(int64_t end, int64_t dims) {
  end = end >= 0 ? end : end + dims;
  CHECK_GT(end, 0);
  return std::min(end, dims);
}

}  // namespace

class SliceV3Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceV3Op);
  SliceV3Op() = default;
  ~SliceV3Op() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_slice_v3_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().slice_v3_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const SliceV3OpConf& conf = op_conf().slice_v3_conf();
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    // DimSliceConf size is 1 less than shape's num of axes,
    // because Slice now don't support dim0 slice,
    CHECK_EQ_OR_RETURN(conf.dim_slice_conf_size() + 1, in->shape().NumAxes());

    DimVector dim_vec(in->shape().NumAxes());
    FOR_RANGE(size_t, i, 0, dim_vec.size()) {
      const int64_t dims = in->shape().At(i);
      CHECK_GT_OR_RETURN(dims, 0);
      if (i == 0) {
        dim_vec[i] = dims;
      } else {
        int64_t begin = FixSliceBegin(conf.dim_slice_conf(i - 1).start(), dims);
        int64_t end = FixSliceEnd(conf.dim_slice_conf(i - 1).end(), dims);
        int64_t stride = conf.dim_slice_conf(i - 1).stride();
        CHECK_NE(begin, end);
        CHECK_NE(stride, 0);
        if (stride > 0) {
          CHECK_LT(begin, end);
        } else {
          CHECK_GT(begin, end);
        }
        int64_t align = (begin > end) ? 1 : -1;
        dim_vec[i] = (end - begin + align) / stride + 1;
      }
    }

    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->CopyMetaFrom(*in);
    out->mut_shape() = Shape(dim_vec);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    SbpSignatureBuilder()
        .PartialSum(input_bns())
        .PartialSum(output_bns())
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

class SliceGradV3Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceGradV3Op);
  SliceGradV3Op() = default;
  ~SliceGradV3Op() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_slice_grad_v3_conf());
    EnrollInputBn("like", false)->set_use_header_only(true);
    EnrollInputBn("dy", false);
    EnrollOutputBn("dx", false);
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().slice_grad_v3_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const auto& conf = op_conf().slice_grad_v3_conf();
    const BlobDesc* like = GetBlobDesc4BnInOp("like");
    // DimSliceConf size is 1 less than shape's num of axes,
    // because Slice now don't support dim0 slice,
    CHECK_EQ_OR_RETURN(conf.dim_slice_conf_size() + 1, like->shape().NumAxes());
    GetBlobDesc4BnInOp("dx")->CopyMetaFrom(*like);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    SbpSignatureBuilder()
        .PartialSum(input_bns())
        .PartialSum(output_bns())
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kSliceV3Conf, SliceV3Op);
REGISTER_OP(OperatorConf::kSliceGradV3Conf, SliceGradV3Op);

}  // namespace oneflow
