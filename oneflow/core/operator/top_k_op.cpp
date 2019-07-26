#include "oneflow/core/operator/top_k_op.h"
#include "oneflow/core/kernel/radix_sort_util.h"

namespace oneflow {

void TopKOp::InitFromOpConf() {
  CHECK(op_conf().has_top_k_conf());
  EnrollInputBn("in", false);
  if (device_type() == DeviceType::kCPU && op_conf().top_k_conf().k() > 1) {
    if (op_conf().top_k_conf().k() > 1) { EnrollFwBufBn("indices"); }
  } else if (device_type() == DeviceType::kGPU) {
    EnrollFwBufBn("temp_storage");
    EnrollFwBufBn("indices");
    EnrollFwBufBn("sorted_in");
    EnrollFwBufBn("sorted_indices");
  }
  EnrollOutputBn("out", false);
}

void TopKOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, int64_t record_piece_size,
                            std::function<void(OpContext*)> EnrollOpCtx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const Shape in_shape = in->shape();
  const int32_t instance_size = in_shape.dim_vec().back();
  const int32_t k = op_conf().top_k_conf().k();
  CHECK_GE(k, 1);
  CHECK_LE(k, instance_size);

  if (device_type() == DeviceType::kCPU) {
    if (k > 1) {
      // fw_buf: indices
      BlobDesc* indices = GetBlobDesc4BnInOp("indices");
      indices->mut_shape() = in_shape;
      indices->set_data_type(DataType::kInt32);
    }
  } else if (device_type() == DeviceType::kGPU) {
    if (instance_size <= 1024 || k == instance_size || k > 128) {
      // fw_buf: indices
      BlobDesc* indices = GetBlobDesc4BnInOp("indices");
      indices->mut_shape() = in_shape;
      indices->set_data_type(DataType::kInt32);
      // fw_buf: sorted_in
      *GetBlobDesc4BnInOp("sorted_in") = *in;
      // fw_buf: sorted_indices
      *GetBlobDesc4BnInOp("sorted_indices") = *indices;
      // fw_buf: temp_storage
      int64_t temp_storage_bytes = InferTempStorageForRadixSort(in_shape.elem_cnt() / instance_size,
                                                                instance_size, in->data_type());
      BlobDesc* temp_storage = GetBlobDesc4BnInOp("temp_storage");
      temp_storage->mut_shape() = Shape({temp_storage_bytes});
      temp_storage->set_data_type(DataType::kChar);
      TopKOpCtx* top_k_op_ctx = new TopKOpCtx(instance_size, k, temp_storage_bytes);
      EnrollOpCtx(top_k_op_ctx);
    }
  } else {
    UNIMPLEMENTED();
  }

  // output
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape().Set(in_shape.NumAxes() - 1, k);
  out->set_data_type(DataType::kInt32);
}

void TopKOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  auto* top_k_op_ctx = static_cast<const TopKOpCtx*>(op_ctx);
  int32_t instance_size = top_k_op_ctx->instance_size_;
  int32_t k = top_k_op_ctx->k_;

  kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  auto* top_k_kernel_conf = kernel_conf->mutable_top_k_conf();
  if (device_type() == DeviceType::kGPU
      && (instance_size <= 1024 || k == instance_size || k > 128)) {
    top_k_kernel_conf->set_temp_storage_bytes(top_k_op_ctx->temp_storage_bytes_);
  } else {
    top_k_kernel_conf->set_temp_storage_bytes(-1);
  }
}

REGISTER_OP(OperatorConf::kTopKConf, TopKOp);

}  // namespace oneflow
