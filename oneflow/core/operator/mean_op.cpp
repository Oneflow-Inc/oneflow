#include "oneflow/core/operator/mean_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void MeanOp::InitFromOpConf() {
  CHECK(op_conf().has_mean_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("fw_tmp");
}

void MeanOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  *out_blob = *in_blob;
  out_blob->mut_shape() = Shape({1});

  BlobDesc* fw_tmp_blob = GetBlobDesc4BnInOp("fw_tmp");
  fw_tmp_blob->mut_shape() = Shape({static_cast<int64_t>(
      GetTmpSizeForReduceSum(in_blob->data_type(), in_blob->shape().elem_cnt()))});
  fw_tmp_blob->set_data_type(DataType::kChar);
}

REGISTER_OP(OperatorConf::kMeanConf, MeanOp);

}  // namespace oneflow
