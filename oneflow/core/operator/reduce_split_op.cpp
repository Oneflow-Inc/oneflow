#include "oneflow/core/operator/reduce_split_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

void ReduceSplitOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_split_conf());
  for (int32_t i = 0; i < op_conf().reduce_split_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
  EnrollInputBn("in", false);
}

const PbMessage& ReduceSplitOp::GetCustomizedConf() const { return op_conf().reduce_split_conf(); }

Maybe<void> ReduceSplitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().reduce_split_conf();
  FOR_RANGE(int32_t, i, 0, conf.out_num()) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    Shape shape(conf.out_shape(i));
    blob_desc->mut_shape() = shape;
    blob_desc->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  }
  return Maybe<void>::Ok();
}

void ReduceSplitOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  ReduceSplitKernelConf* reduce_split_conf = kernel_conf->mutable_reduce_split_conf();
  int64_t offset = 0;
  for (int32_t i = 0; i < op_conf().reduce_split_conf().out_num(); ++i) {
    reduce_split_conf->mutable_data_offset()->Add(offset);
    offset += RtBlobDesc(*(GetBlobDesc4BnInOp(output_bns().Get(i)))).ByteSizeOfBlobBody();
  }
  const int64_t data_type_byte_size =
      static_cast<int64_t>(GetSizeOfDataType(GetBlobDesc4BnInOp(SoleIbn())->data_type()));
  CHECK_EQ(offset % data_type_byte_size, 0);
  const int64_t out_blob_elem_cnt_sum =
      RoundUp(offset / data_type_byte_size, parallel_ctx->parallel_num());
  const int64_t in_blob_elem_cnt = GetBlobDesc4BnInOp(SoleIbn())->shape().elem_cnt();
  CHECK_EQ(out_blob_elem_cnt_sum, in_blob_elem_cnt);
}

Maybe<void> ReduceSplitOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  CHECK_EQ_OR_RETURN(*HasBatchDim4BnInOp("in"), false);
  for (const auto& ibn : input_bns()) { *HasBatchDim4BnInOp(ibn) = false; }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReduceSplitConf, ReduceSplitOp);

}  // namespace oneflow
