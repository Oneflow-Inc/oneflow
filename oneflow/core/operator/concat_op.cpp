#include "oneflow/core/operator/concat_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_concat_conf());

  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ConcatOp::GetCustomizedConf() const { return op_conf().concat_conf(); }

void ConcatOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const ConcatOpConf& conf = op_conf().concat_conf();
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  std::vector<int64_t> out_dim_vec = in_0_blob_desc->shape().dim_vec();
  int32_t concat_axis = conf.axis();
  if (concat_axis < 0) { concat_axis += out_dim_vec.size(); }
  CHECK_GE(concat_axis, 1);
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const BlobDesc* in_i_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    for (int64_t j = 0; j < in_i_blob_desc->shape().NumAxes(); ++j) {
      if (j == concat_axis) {
        out_dim_vec[j] += in_i_blob_desc->shape().At(j);
      } else {
        CHECK_EQ(out_dim_vec[j], in_i_blob_desc->shape().At(j));
      }
    }
    CHECK_EQ(in_i_blob_desc->data_type(), in_0_blob_desc->data_type());
    CHECK_EQ(in_i_blob_desc->has_data_id_field(), in_0_blob_desc->has_data_id_field());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_0_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
}

void ConcatOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const ConcatOpConf& conf = op_conf().concat_conf();
  const int axis = conf.axis();
  const int num_axes = LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes();
  for (int i = 0; i < num_axes; ++i) {
    if (i == axis) { continue; }
    SbpSignatureBuilder()
        .Split(input_bns(), i)
        .Split(output_bns(), i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
}

REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

}  // namespace oneflow
