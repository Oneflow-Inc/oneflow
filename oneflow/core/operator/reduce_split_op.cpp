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

void ReduceSplitOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().reduce_split_conf();
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    FOR_RANGE(int32_t, i, 0, conf.out_num()) {
      Shape shape(conf.out_shape(i));
      CHECK_EQ(GetBlobDesc4BnInOp(output_bns().Get(i))->shape().elem_cnt(), shape.elem_cnt());
      GetBlobDesc4BnInOp(output_bns().Get(i))->mut_shape() = shape;
    }
  }
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

REGISTER_OP(OperatorConf::kReduceSplitConf, ReduceSplitOp);

}  // namespace oneflow
