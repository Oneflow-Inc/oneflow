#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/radix_sort_op_util.h"

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

  void InitFromOpConf() override {
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
  const PbMessage& GetCustomizedConf() const override { return this->op_conf().top_k_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const int32_t instance_size = in->shape().dim_vec().back();
    const int32_t k = op_conf().top_k_conf().k();
    CHECK_GE_OR_RETURN(k, 1);
    // temp solution: we allow k > instance_size
    // CHECK_LE_OR_RETURN(k, instance_size);
    if (device_type() == DeviceType::kCPU) {
      if (k > 1) {
        // tmp: indices
        BlobDesc* indices = GetBlobDesc4BnInOp("indices");
        *indices = *in;
        indices->set_data_type(DataType::kInt32);
      }
    } else if (device_type() == DeviceType::kGPU) {
      if (k > 128) {
        // tmp: indices
        BlobDesc* indices = GetBlobDesc4BnInOp("indices");
        *indices = *in;
        indices->set_data_type(DataType::kInt32);
        // tmp: sorted_in
        *GetBlobDesc4BnInOp("sorted_in") = *in;
        // tmp: sorted_indices
        *GetBlobDesc4BnInOp("sorted_indices") = *indices;
        // tmp: temp_storage
        int64_t temp_storage_bytes = InferTempStorageForSortingPairsDescendingAtCompile(
            in->shape().elem_cnt() / instance_size, instance_size, in->data_type());
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
    out->mut_shape().Set(in->shape().NumAxes() - 1, std::min(k, instance_size));
    out->set_data_type(DataType::kInt32);

    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
    return Maybe<void>::Ok();
  }
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf* kernel_conf, const OpContext* op_ctx) const override {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
    if (device_type() == DeviceType::kGPU && op_conf().top_k_conf().k() > 128) {
      auto* top_k_op_ctx = static_cast<const TopKOpCtx*>(op_ctx);
      kernel_conf->mutable_top_k_conf()->set_temp_storage_bytes(top_k_op_ctx->temp_storage_bytes_);
    }
  }
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kTopKConf, TopKOp);

}  // namespace oneflow
