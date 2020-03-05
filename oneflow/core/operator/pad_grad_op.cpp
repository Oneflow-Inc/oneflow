#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PadGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PadGradOp);
  PadGradOp() = default;
  ~PadGradOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_pad_grad_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().pad_grad_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    out_blob_desc->set_data_type(in_blob_desc->data_type());

    DimVector out_shape(in_blob_desc->shape().dim_vec());
    const int64_t ndims = op_conf().pad_grad_conf().paddings_size() / 2;
    const int64_t offset = in_blob_desc->shape().NumAxes() - ndims;
    for (int64_t i = 0; i < in_blob_desc->shape().NumAxes(); ++i) {
        if (i >= offset){
          int64_t j = i - offset;
          int64_t padding_before = GetPbRfFromCustomizedConf<int32_t>("paddings").Get(2*j);
          int64_t padding_after = GetPbRfFromCustomizedConf<int32_t>("paddings").Get(2*j + 1);
          out_shape[i] = in_blob_desc->shape().At(i) - padding_before - padding_after;
        }else{
          out_shape[i] = in_blob_desc->shape().At(i);
        }
    }
    out_blob_desc->mut_shape() = Shape(out_shape);
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
        .MakeSplitSignatureListBuilder(
            JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kPadGradConf, PadGradOp);

}  // namespace oneflow
