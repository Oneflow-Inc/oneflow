#include "oneflow/core/operator/operator.h"

namespace oneflow {

class DynamicReshapeOp final : public Operator {
 public:
  void InitFromOpConf() {
    CHECK(op_conf().has_dynamic_reshape_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
  }
  const PbMessage& GetCustomizedConf() const { return op_conf().dynamic_reshape_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
    const DynamicReshapeOpConf& conf = op_conf().dynamic_reshape_conf();
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    Shape out_shape(conf.shape());
    if (out_shape.NumAxes() == 1) {
      CHECK_EQ_OR_RETURN(-1, out_shape.At(0));
      out_shape = Shape({in->shape().Count(0)});
    } else {
      int32_t inferred_axis = -1;
      std::vector<int64_t> tmp_dim_vec;
      for (int32_t i = 0; i < out_shape.NumAxes(); ++i) {
        if (out_shape.At(i) == -1) {
          CHECK_EQ_OR_RETURN(-1, inferred_axis);
          inferred_axis = i;
        } else {
          CHECK_GT_OR_RETURN(out_shape.At(i), 0);
          tmp_dim_vec.push_back(out_shape.At(i));
        }
      }
      CHECK_GE_OR_RETURN(inferred_axis, 0);
      out_shape.Set(inferred_axis, in->shape().elem_cnt() / Shape(tmp_dim_vec).elem_cnt());
    }
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    out->mut_shape() = std::move(out_shape);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
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
    EnrollInputBn("like", false)->set_use_header_only(true);
  }
  const PbMessage& GetCustomizedConf() const { return op_conf().dynamic_reshape_like_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("x")->shape().elem_cnt(),
                       GetBlobDesc4BnInOp("like")->shape().elem_cnt());
    GetBlobDesc4BnInOp("y")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
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
