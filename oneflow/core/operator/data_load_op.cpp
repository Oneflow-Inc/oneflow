#include "oneflow/core/operator/data_load_op.h"

namespace oneflow {

void DataLoadOp::InitFromOpConf() {
  CHECK(op_conf().has_data_load_conf());
  EnrollOutputBn("out", false);
}

const PbMessage& DataLoadOp::GetCustomizedConf() const { return op_conf().data_load_conf(); }

void DataLoadOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                int64_t record_piece_size) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  CHECK_EQ(record_piece_size % parallel_ctx->parallel_num(), 0);
  out_blob_desc->mut_shape() = Shape({record_piece_size / parallel_ctx->parallel_num()});
  out_blob_desc->set_data_type(kOFRecord);
}

// void DataLoadOp::VirtualGenKernelConf(
//     std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
//     const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
//   int64_t device_piece_size = GetBlobDesc4BnInOp("out")->shape().At(0);
//   kernel_conf->mutable_record_load_conf()->set_device_piece_size(device_piece_size);
// }

REGISTER_CPU_OP(OperatorConf::kDataLoadConf, DataLoadOp);

}  // namespace oneflow
