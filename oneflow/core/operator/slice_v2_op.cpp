#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class SliceV2Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceV2Op);
  SliceV2Op() = default;
  ~SliceV2Op() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_slice_v2_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
    EnrollTmpBn("out_to_in_offset");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().slice_v2_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    auto ret = InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx);
    BlobDesc* offset_blob_desc = GetBlobDesc4BnInOp("out_to_in_offset");
    *offset_blob_desc = *GetBlobDesc4BnInOp("out");
    offset_blob_desc->set_data_type(DataType::kInt64);
    return ret;
  }
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    const SliceV2OpConf& conf = op_conf().slice_v2_conf();
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
    CHECK_EQ_OR_RETURN(conf.dim_slice_conf_size(), in_blob_desc->shape().NumAxes());
    DimVector shape_vec(in_blob_desc->shape().NumAxes());
    FOR_RANGE(size_t, i, 0, conf.dim_slice_conf_size()) {
      const int64_t dim_len = in_blob_desc->shape().At(i);
      if (dim_len > 0) {
        const DimSliceConf& dim_slice_conf = conf.dim_slice_conf(i);
        int64_t start = dim_slice_conf.has_start() ? dim_slice_conf.start() : 0;
        if (start < 0) { start += dim_len; }
        CHECK_GE_OR_RETURN(start, 0);
        CHECK_LT_OR_RETURN(start, dim_len);
        int64_t end = dim_slice_conf.has_end() ? dim_slice_conf.end() : dim_len;
        if (end < 0) { end += dim_len; }
        if (end > dim_len) { end = dim_len; }
        CHECK_LT_OR_RETURN(start, end);
        int64_t step = dim_slice_conf.stride();
        CHECK_GT_OR_RETURN(step, 0);
        shape_vec[i] = (end - 1 - start) / step + 1;
      } else {
        shape_vec[i] = 0;
      }
    }

    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    *out_blob_desc = *in_blob_desc;
    out_blob_desc->mut_shape() = Shape(shape_vec);
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

REGISTER_OP(OperatorConf::kSliceV2Conf, SliceV2Op);

}  // namespace oneflow
