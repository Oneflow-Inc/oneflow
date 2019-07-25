#include "oneflow/core/operator/top_k_op.h"
#include "oneflow/core/kernel/radix_sort_util.h"

namespace oneflow {

void TopKOp::InitFromOpConf() {
  CHECK(op_conf().has_top_k_conf());
  EnrollInputBn("in", false);
  if (device_type() == DeviceType::kCPU && op_conf().top_k_conf().k() > 1) {
    if (op_conf().top_k_conf().k() > 1) { EnrollFwBufBn("cpu_indices"); }
  } else if (device_type() == DeviceType::kGPU) {
    EnrollFwBufBn("temp_storage");
    EnrollFwBufBn("gpu_indices");
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
  CHECK_LE(in_shape.elem_cnt(), GetMaxVal<int32_t>());
  const int32_t instance_size = in_shape.dim_vec().back();
  const TopKOpConf& conf = op_conf().top_k_conf();
  CHECK_GE(conf.k(), 1);
  CHECK_LE(conf.k(), instance_size);

  if (device_type() == DeviceType::kCPU && conf.k() > 1) {
    // fw_buf: cpu_indices
    BlobDesc* cpu_indices = GetBlobDesc4BnInOp("cpu_indices");
    cpu_indices->mut_shape() = Shape(in_shape);
    cpu_indices->set_data_type(DataType::kInt32);
  }
  if (device_type() == DeviceType::kGPU
      && (instance_size <= 1000 || conf.k() == instance_size || conf.k() > 512)) {
    // fw_buf: gpu_indices
    BlobDesc* gpu_indices = GetBlobDesc4BnInOp("gpu_indices");
    gpu_indices->mut_shape() = Shape(in_shape);
    gpu_indices->set_data_type(DataType::kInt32);

    // fw_buf: sorted_in
    *GetBlobDesc4BnInOp("sorted_in") = *in;

    // fw_buf: sorted_indices
    *GetBlobDesc4BnInOp("sorted_indices") = *gpu_indices;

    // fw_buf: temp_storage
    int64_t temp_storage_bytes = InferTempStorageForRadixSort(
        in_shape.Count(0, in_shape.NumAxes() - 1), in_shape.At(1), in->data_type());
    BlobDesc* temp_storage = GetBlobDesc4BnInOp("temp_storage");
    temp_storage->set_data_type(DataType::kChar);
    temp_storage->mut_shape() = Shape({temp_storage_bytes});
    TopKOpCtx* top_k_op_ctx = new TopKOpCtx(temp_storage_bytes);
    EnrollOpCtx(top_k_op_ctx);
  }

  // output
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape().Set(in_shape.NumAxes() - 1, conf.k());
  out->set_data_type(DataType::kInt32);
}

void TopKOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const TopKOpConf& top_k_op_conf = op_conf().top_k_conf();
  auto* top_k_kernel_conf = kernel_conf->mutable_top_k_conf();
  const int32_t instance_size = in->shape().dim_vec().back();

  kernel_conf->set_data_type(in->data_type());
  if (device_type() == DeviceType::kGPU
      && (instance_size <= 1000 || top_k_op_conf.k() == instance_size || top_k_op_conf.k() > 512)) {
    auto* top_k_op_ctx = static_cast<const TopKOpCtx*>(op_ctx);
    top_k_kernel_conf->set_temp_storage_bytes(top_k_op_ctx->GetTempStorageBytes());
  } else {
    top_k_kernel_conf->set_temp_storage_bytes(-1);
  }
}

REGISTER_OP(OperatorConf::kTopKConf, TopKOp);

}  // namespace oneflow
