#include "oneflow/core/operator/batch_gather_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void BatchGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_batch_gather_conf());
  EnrollInputBn("in");
  EnrollInputBn("indices", false);
  EnrollOutputBn("out");
}

const PbMessage& BatchGatherOp::GetCustomizedConf() const { return op_conf().batch_gather_conf(); }

Maybe<void> BatchGatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK_GT_OR_RETURN(indices->shape().NumAxes(), 0);
  CHECK_OR_RETURN(IsIntegralDataType(indices->data_type()));
  const std::vector<int64_t>& in_dim_vec = in->shape().dim_vec();
  const std::vector<int64_t>& indices_dim_vec = indices->shape().dim_vec();
  CHECK_LE_OR_RETURN(indices_dim_vec.size(), in_dim_vec.size());
  FOR_RANGE(int64_t, i, 0, indices_dim_vec.size() - 1) {
    CHECK_EQ_OR_RETURN(indices_dim_vec.at(i), in_dim_vec.at(i));
  }
  // out
  std::vector<int64_t> out_dim_vec(indices_dim_vec);
  out_dim_vec.insert(out_dim_vec.end(), in_dim_vec.cbegin() + indices_dim_vec.size(),
                     in_dim_vec.cend());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

void BatchGatherOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t indices_num_axes = LogicalBlobDesc4Ibn("indices").shape().NumAxes();
  if (indices_num_axes > 1) {
    FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
      SbpSignatureBuilder()
          .Split("indices", i)
          .Split("in", i)
          .Split("out", i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_OP(OperatorConf::kBatchGatherConf, BatchGatherOp);

}  // namespace oneflow
