#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct TopKOpCtx : public OpContext {
#ifdef WITH_CUDA
  TopKOpCtx(int32_t temp_storage_bytes) : temp_storage_bytes_(temp_storage_bytes) {}
  int32_t temp_storage_bytes_;
#endif
};

class TopKOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TopKOp);
  TopKOp() = default;
  ~TopKOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return this->op_conf().top_k_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*, const OpContext*) const override;
};

void TopKOp::InitFromOpConf() {
  CHECK(op_conf().has_top_k_conf());
  EnrollInputBn("in", false);
  const int32_t k = op_conf().top_k_conf().k();
  if (device_type() == DeviceType::kCPU) {
    if (k > 1) { EnrollTmpBn("indices"); }
  } else if (device_type() == DeviceType::kGPU) {
    if (k > 128) {
      EnrollTmpBn("indices");
      EnrollTmpBn("sorted_in");
      EnrollTmpBn("sorted_indices");
      EnrollTmpBn("temp_storage");
    }
  }
  EnrollOutputBn("out", false);
}

Maybe<void> TopKOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext*, const SbpSignature*,
                                   std::function<void(OpContext*)> EnrollOpCtx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const int32_t instance_size = in->shape().dim_vec().back();
  const int32_t k = op_conf().top_k_conf().k();
  CHECK_GE(k, 1);
  CHECK_LE(k, instance_size);

  if (device_type() == DeviceType::kCPU) {
    if (k > 1) {
      // fw_buf: indices
      BlobDesc* indices = GetBlobDesc4BnInOp("indices");
      *indices = *in;
      indices->set_data_type(DataType::kInt32);
    }
  } else if (device_type() == DeviceType::kGPU) {
    if (k > 128) {
      // fw_buf: indices
      BlobDesc* indices = GetBlobDesc4BnInOp("indices");
      *indices = *in;
      indices->set_data_type(DataType::kInt32);
      // fw_buf: sorted_in
      *GetBlobDesc4BnInOp("sorted_in") = *in;
      // fw_buf: sorted_indices
      *GetBlobDesc4BnInOp("sorted_indices") = *indices;
      // fw_buf: temp_storage
      // TODO: infer temp_storage_byte by cub instead of setting it to 100
      // int64_t temp_storage_bytes = InferTempStorageForSortingPairsDescendingAtCompile(
      //     in->shape().elem_cnt() / instance_size, instance_size, in->data_type());
      int64_t temp_storage_bytes = 100;
      BlobDesc* temp_storage = GetBlobDesc4BnInOp("temp_storage");
      temp_storage->mut_shape() = Shape({temp_storage_bytes});
      temp_storage->set_data_type(DataType::kChar);
      TopKOpCtx* top_k_op_ctx = new TopKOpCtx(temp_storage_bytes);
      EnrollOpCtx(top_k_op_ctx);
    }
  } else {
    UNIMPLEMENTED();
  }

  // output
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape().Set(in->shape().NumAxes() - 1, k);
  out->set_data_type(DataType::kInt32);

  return Maybe<void>::Ok();
}

void TopKOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  if (device_type() == DeviceType::kGPU && op_conf().top_k_conf().k() > 128) {
    auto* top_k_op_ctx = static_cast<const TopKOpCtx*>(op_ctx);
    kernel_conf->mutable_top_k_conf()->set_temp_storage_bytes(top_k_op_ctx->temp_storage_bytes_);
  }
}

REGISTER_OP(OperatorConf::kTopKConf, TopKOp);

}  // namespace oneflow
