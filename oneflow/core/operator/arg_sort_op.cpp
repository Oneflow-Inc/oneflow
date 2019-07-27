#include "oneflow/core/operator/arg_sort_op.h"
#include "oneflow/core/operator/radix_sort_op_util.h"

namespace oneflow {

void ArgSortOp::InitFromOpConf() {
  CHECK(op_conf().has_arg_sort_conf());
  EnrollInputBn("in", false);
  if (device_type() == DeviceType::kGPU) {
    EnrollFwBufBn("indices");
    EnrollFwBufBn("sorted_in");
    EnrollFwBufBn("temp_storage");
  }
  EnrollOutputBn("out" /*sorted_indices */, false);
}

const PbMessage& ArgSortOp::GetCustomizedConf() const { return this->op_conf().arg_sort_conf(); }

void ArgSortOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext*, int64_t record_piece_size,
                               std::function<void(OpContext*)> EnrollOpCtx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const int32_t instance_size = in->shape().dim_vec().back();
  const int32_t instance_num = in->shape().elem_cnt() / instance_size;

  if (device_type() == DeviceType::kGPU) {
    // fw_buf: indices
    BlobDesc* indices = GetBlobDesc4BnInOp("indices");
    *indices = *in;
    indices->set_data_type(DataType::kInt32);
    indices->set_has_dim0_valid_num_field(in->has_dim0_valid_num_field());
    indices->mut_dim0_inner_shape() = Shape({1, in->shape().At(0)});
    indices->set_has_instance_shape_field(in->has_instance_shape_field());

    // fw_buf: temp_storage
    int32_t temp_storage_bytes = -1;
    if (op_conf().arg_sort_conf().dir() == "ASCENDING") {
      temp_storage_bytes = InferTempStorageForSortingPairsAscendingAtCompile(
          instance_num, instance_size, in->data_type());
    } else if (op_conf().arg_sort_conf().dir() == "DESCENDING") {
      temp_storage_bytes = InferTempStorageForSortingPairsDescendingAtCompile(
          instance_num, instance_size, in->data_type());
    } else {
      UNIMPLEMENTED();
    }
    BlobDesc* temp_storage = GetBlobDesc4BnInOp("temp_storage");
    temp_storage->set_data_type(DataType::kChar);
    temp_storage->mut_shape() = Shape({temp_storage_bytes});
    ArgSortOpCtx* arg_sort_op_ctx = new ArgSortOpCtx(temp_storage_bytes);
    EnrollOpCtx(arg_sort_op_ctx);

    // fw_buf: sorted_in
    *GetBlobDesc4BnInOp("sorted_in") = *in;
  }

  // output
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->set_data_type(DataType::kInt32);
}

void ArgSortOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  if (device_type() == DeviceType::kGPU) {
    auto* arg_sort_op_ctx = static_cast<const ArgSortOpCtx*>(op_ctx);
    kernel_conf->mutable_arg_sort_conf()->set_temp_storage_bytes(
        arg_sort_op_ctx->temp_storage_bytes_);
  }
}

REGISTER_OP(OperatorConf::kArgSortConf, ArgSortOp);

}  // namespace oneflow
