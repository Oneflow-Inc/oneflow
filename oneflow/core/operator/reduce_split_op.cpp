#include "oneflow/core/operator/reduce_split_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

void ReduceSplitOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_split_conf());
  EnrollInputBn("in", false);
  EnrollRepeatedOutputBn("out", false);
}

const PbMessage& ReduceSplitOp::GetCustomizedConf() const { return op_conf().reduce_split_conf(); }

Maybe<void> ReduceSplitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().reduce_split_conf();
  FOR_RANGE(int32_t, i, 0, conf.out_size()) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    Shape shape(conf.out_shape(i));
    blob_desc->mut_shape() = shape;
    blob_desc->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  }

  // Check valid (but can be delete)
  {
    CHECK_EQ(conf.out_size(), conf.data_offset_size());
    int64_t offset = 0;
    for (int32_t i = 0; i < conf.out_size(); ++i) {
      CHECK_EQ(offset, conf.data_offset(i));
      offset += RtBlobDesc(*(GetBlobDesc4BnInOp(output_bns().Get(i)))).AlignedByteSizeOfBlobBody();
    }
    const int64_t data_type_byte_size =
        static_cast<int64_t>(GetSizeOfDataType(GetBlobDesc4BnInOp(SoleIbn())->data_type()));
    CHECK_EQ(offset % data_type_byte_size, 0);
    const int64_t out_blob_elem_cnt_sum =
        RoundUp(offset / data_type_byte_size, parallel_ctx->parallel_num());
    const int64_t in_blob_elem_cnt = GetBlobDesc4BnInOp(SoleIbn())->shape().elem_cnt();
    CHECK_EQ(out_blob_elem_cnt_sum, in_blob_elem_cnt);
  }

  return Maybe<void>::Ok();
}

Maybe<void> ReduceSplitOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_EQ_OR_RETURN(BatchAxis4BnInOp("in")->has_value(), false);
  for (const auto& obn : output_bns()) { BatchAxis4BnInOp(obn)->clear_value(); }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReduceSplitConf, ReduceSplitOp);

}  // namespace oneflow
