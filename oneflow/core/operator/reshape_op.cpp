#include "oneflow/core/operator/reshape_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReshapeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_const_inplace_ibn("in");
}

const PbMessage& ReshapeOp::GetCustomizedConf() const { return op_conf().reshape_conf(); }

Maybe<void> ReshapeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;

  const ReshapeOpConf& conf = op_conf().reshape_conf();
  CHECK_GE_OR_RETURN(conf.shape().dim_size(), 1);
  std::vector<int64_t> dim_vec;
  int32_t begin_index = 0;
  if (conf.has_dim0_in_shape()) {
    // infer dim0 as split(0)
    if (conf.shape().dim(0) > 0) {
      CHECK_GE_OR_RETURN(conf.shape().dim(0), parallel_ctx->parallel_num());
      BalancedSplitter splitter(conf.shape().dim(0), parallel_ctx->parallel_num());
      dim_vec.push_back(splitter.At(parallel_ctx->parallel_id()).size());
    } else {
      dim_vec.push_back(-1);
    }
    begin_index = 1;
  } else {
    dim_vec.push_back(in_blob_desc->shape().At(0));
  }
  for (int32_t i = begin_index; i < conf.shape().dim_size(); ++i) {
    dim_vec.push_back(conf.shape().dim(i));
  }
  int32_t dim_cnt_need_infer = 0;
  int32_t dim_index_need_infer = -1;
  int64_t elem_cnt = 1;
  for (int32_t i = 0; i < dim_vec.size(); ++i) {
    if (dim_vec[i] == -1) {
      ++dim_cnt_need_infer;
      dim_index_need_infer = i;
    } else {
      CHECK_GT_OR_RETURN(dim_vec[i], 0);
      elem_cnt *= dim_vec[i];
    }
  }
  CHECK_LE_OR_RETURN(dim_cnt_need_infer, 1);
  if (dim_cnt_need_infer == 1) {
    dim_vec[dim_index_need_infer] = in_blob_desc->shape().elem_cnt() / elem_cnt;
  }
  out_blob_desc->mut_shape() = Shape(dim_vec);
  CHECK_EQ_OR_RETURN(out_blob_desc->shape().elem_cnt(), in_blob_desc->shape().elem_cnt());
  return Maybe<void>::Ok();
}

void ReshapeOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kReshapeConf, ReshapeOp);

}  // namespace oneflow
