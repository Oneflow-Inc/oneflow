#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LocalNonzeroOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalNonzeroOp);
  LocalNonzeroOp() = default;
  ~LocalNonzeroOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_local_nonzero_conf());
    EnrollInputBn("in", false);
    if (this->device_type() == DeviceType::kGPU) {
      EnrollTmpBn("shape");
      EnrollTmpBn("num_nonzero");
    }
    EnrollOutputBn("out", false);
  }

  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().local_nonzero_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const int64_t elem_cnt = in->shape().elem_cnt();
    if (this->device_type() == DeviceType::kGPU) {
      // data tmp: shape
      BlobDesc* shape = GetBlobDesc4BnInOp("shape");
      shape->mut_shape() = Shape({in->shape().NumAxes()});
      shape->set_data_type(DataType::kInt64);
      // data tmp: num_nonzero
      BlobDesc* num_nonzero = GetBlobDesc4BnInOp("num_nonzero");
      num_nonzero->mut_shape() = Shape({1});
      num_nonzero->set_data_type(DataType::kInt64);
    }
    // output
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->mut_shape() = Shape({elem_cnt, in->shape().NumAxes()});
    out->set_data_type(DataType::kInt32);
    out->set_has_dim0_valid_num_field(true);
    out->mut_dim0_inner_shape() = Shape({1, elem_cnt});
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  }
};

REGISTER_OP(OperatorConf::kLocalNonzeroConf, LocalNonzeroOp);

}  // namespace oneflow
