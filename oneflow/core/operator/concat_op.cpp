#include "oneflow/core/operator/concat_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_concat_conf());

  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ConcatOp::GetCustomizedConf() const { return op_conf().concat_conf(); }

Maybe<void> ConcatOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ConcatOpConf& conf = op_conf().concat_conf();
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  DimVector out_dim_vec = in_0_blob_desc->shape().dim_vec();
  int32_t concat_axis = FixAxis(conf.axis(), out_dim_vec.size());
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const BlobDesc* in_i_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    for (int64_t j = 0; j < in_i_blob_desc->shape().NumAxes(); ++j) {
      if (j == concat_axis) {
        out_dim_vec[j] += in_i_blob_desc->shape().At(j);
      } else {
        CHECK_EQ_OR_RETURN(out_dim_vec[j], in_i_blob_desc->shape().At(j));
      }
    }
    CHECK_EQ_OR_RETURN(in_i_blob_desc->data_type(), in_0_blob_desc->data_type());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_0_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> ConcatOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const ConcatOpConf& conf = op_conf().concat_conf();
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes();
  const int32_t axis = FixAxis(conf.axis(), num_axes);
  for (int64_t i = 0; i < num_axes; ++i) {
    if (i == axis) { continue; }
    SbpSignatureBuilder()
        .Split(input_bns(), i)
        .Split(output_bns(), i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

int32_t ConcatOp::FixAxis(const int32_t axis, const int64_t num_axes) const {
  int32_t ret = axis;
  if (axis < 0) { ret += num_axes; }
  CHECK_GE(ret, 0);
  CHECK_LT(ret, num_axes);
  return ret;
}

REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

}  // namespace oneflow
