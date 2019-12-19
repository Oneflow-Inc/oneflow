#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LocalScatterNdUpdateOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalScatterNdUpdateOp);
  LocalScatterNdUpdateOp() = default;
  ~LocalScatterNdUpdateOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_local_scatter_nd_update_conf());
    EnrollInputBn("in");
    EnrollInputBn("indices", false);
    EnrollInputBn("updates");
    if (this->device_type() == DeviceType::kGPU) { EnrollTmpBn("shape"); }
    EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
  }

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().local_scatter_nd_update_conf();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    ScatterNdUpdateKernelConf* conf = kernel_conf->mutable_scatter_nd_update_conf();
    conf->set_idx_type(GetBlobDesc4BnInOp("indices")->data_type());
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input: in, indices, updates
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
    const BlobDesc* updates = GetBlobDesc4BnInOp("updates");
    CHECK_EQ_OR_RETURN(in->is_tensor_list(), false);
    CHECK_EQ_OR_RETURN(indices->is_tensor_list(), false);
    CHECK_EQ_OR_RETURN(updates->is_tensor_list(), false);
    OF_CHECK_EQ(in->data_type(), updates->data_type());
    OF_CHECK(IsIntegralDataType(indices->data_type()));

    int64_t segm_dims = indices->shape().At(indices->shape().NumAxes() - 1);
    OF_CHECK_LE(segm_dims, in->shape().NumAxes());
    FOR_RANGE(int64_t, i, 0, segm_dims) {
      OF_CHECK_EQ(indices->shape().At(i), updates->shape().At(i));
    }
    FOR_RANGE(int64_t, i, 0, in->shape().NumAxes() - segm_dims) {
      OF_CHECK_EQ(in->shape().At(i + segm_dims),
                  updates->shape().At(i + indices->shape().NumAxes() - 1));
    }
    *GetBlobDesc4BnInOp("out") = *in;
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input: in, indices, updates
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
    const BlobDesc* updates = GetBlobDesc4BnInOp("updates");
    CHECK_EQ_OR_RETURN(in->is_tensor_list(), false);
    CHECK_EQ_OR_RETURN(indices->is_tensor_list(), false);
    CHECK_EQ_OR_RETURN(updates->is_tensor_list(), false);
    OF_CHECK_EQ(in->data_type(), updates->data_type());
    OF_CHECK(IsIntegralDataType(indices->data_type()));

    int64_t segm_dims = indices->shape().At(indices->shape().NumAxes() - 1);
    OF_CHECK_LE(segm_dims, in->shape().NumAxes());
    FOR_RANGE(int64_t, i, 0, segm_dims) {
      OF_CHECK_EQ(indices->shape().At(i), updates->shape().At(i));
    }
    FOR_RANGE(int64_t, i, 0, in->shape().NumAxes() - segm_dims) {
      OF_CHECK_EQ(in->shape().At(i + segm_dims),
                  updates->shape().At(i + indices->shape().NumAxes() - 1));
    }

    BlobDesc* shape = GetBlobDesc4BnInOp("shape");
    if (shape) {
      shape->mut_shape() = Shape({in->shape().NumAxes()});
      shape->set_data_type(DataType::kInt64);
    }

    *GetBlobDesc4BnInOp("out") = *in;
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("in", 0)
        .Split("indices", 0)
        .Split("updates", 0)
        .Split("out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }

  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    SbpSignatureList sbp_sig_list;
    JUST(GetSbpSignatures(&sbp_sig_list));
    CHECK_EQ(sbp_sig_list.sbp_signature_size(), 1);
    *sbp_signature = sbp_sig_list.sbp_signature(0);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLocalScatterNdUpdateConf, LocalScatterNdUpdateOp);

}  // namespace oneflow
