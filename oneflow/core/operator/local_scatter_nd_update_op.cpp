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
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().local_scatter_nd_update_conf();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    // LocalScatterNdUpdateConf* conf = kernel_conf->mutable_local_scatter_nd_update_conf();
    // conf->set_value_type(GetBlobDesc4BnInOp("in")->data_type());
    // conf->set_indices_type(GetBlobDesc4BnInOp("indices")->data_type());
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input: in, indices, updates
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
    const BlobDesc* updates = GetBlobDesc4BnInOp("updates");
    CHECK_EQ(in->data_type(), updates->data_type());
    CHECK(IsIntegralDataType(indices->data_type()));
    const auto indices_dim_vec = indices->shape().dim_vec();
    if (this->device_type() == DeviceType::kGPU) {
      // datatmp
      BlobDesc* shape = GetBlobDesc4BnInOp("shape");
      shape->mut_shape() = Shape({in->shape().NumAxes()});
      shape->set_data_type(DataType::kInt64);
    }

    // output
    *GetBlobDesc4BnInOp("out") = *in;
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLocalScatterNdUpdateConf, LocalScatterNdUpdateOp);

}  // namespace oneflow
