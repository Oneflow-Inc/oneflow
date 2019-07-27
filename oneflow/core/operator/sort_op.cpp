#include "oneflow/core/operator/sort_op.h"
#include "oneflow/core/operator/radix_sort_op_util.h"

namespace oneflow {

void SortOp::InitFromOpConf() {
  CHECK(op_conf().has_sort_conf());
  EnrollInputBn("in", false);
  if (device_type() == DeviceType::kGPU) { EnrollFwBufBn("temp_storage"); }
  EnrollOutputBn("out", false);
}

const PbMessage& SortOp::GetCustomizedConf() const { return this->op_conf().sort_conf(); }

void SortOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, int64_t record_piece_size,
                            std::function<void(OpContext*)> EnrollOpCtx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const int32_t instance_size = in->shape().dim_vec().back();
  const int32_t instance_num = in->shape().elem_cnt() / instance_size;

  if (device_type() == DeviceType::kGPU) {
    // fw_buf: temp_storage
    int32_t temp_storage_bytes = -1;
    if (op_conf().sort_conf().dir() == "ASCENDING") {
      temp_storage_bytes = InferTempStorageForSortingKeysAscendingAtCompile(
          instance_num, instance_size, in->data_type());
    } else if (op_conf().sort_conf().dir() == "DESCENDING") {
      temp_storage_bytes = InferTempStorageForSortingKeysDescendingAtCompile(
          instance_num, instance_size, in->data_type());
    } else {
      UNIMPLEMENTED();
    }
    BlobDesc* temp_storage = GetBlobDesc4BnInOp("temp_storage");
    temp_storage->mut_shape() = Shape({temp_storage_bytes});
    temp_storage->set_data_type(DataType::kChar);
    SortOpCtx* sort_op_ctx = new SortOpCtx(temp_storage_bytes);
    EnrollOpCtx(sort_op_ctx);
  }

  // output
  *GetBlobDesc4BnInOp("out") = *in;
}

void SortOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  if (device_type() == DeviceType::kGPU) {
    auto* sort_op_ctx = static_cast<const SortOpCtx*>(op_ctx);
    kernel_conf->mutable_sort_conf()->set_temp_storage_bytes(sort_op_ctx->temp_storage_bytes_);
  }
}

REGISTER_OP(OperatorConf::kSortConf, SortOp);

}  // namespace oneflow
