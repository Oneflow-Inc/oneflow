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
    if (this->device_type() == DeviceType::kGPU) { EnrollTmpBn("shape"); }
    EnrollOutputBn("out", false);
    EnrollOutputBn("num_nonzero", false);
  }

  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().local_nonzero_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const int64_t elem_cnt = in->shape().elem_cnt();
    if (this->device_type() == DeviceType::kGPU) {
      // data tmp: shape
      BlobDesc* shape = GetBlobDesc4BnInOp("shape");
      shape->mut_shape() = Shape({in->shape().NumAxes()});
      shape->set_data_type(DataType::kInt64);
    }
    // output
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->mut_shape() = Shape({elem_cnt, in->shape().NumAxes()});
    out->set_data_type(DataType::kInt32);
    // output: num_nonzero
    BlobDesc* num_nonzero = GetBlobDesc4BnInOp("num_nonzero");
    num_nonzero->mut_shape() = Shape({1});
    num_nonzero->set_data_type(DataType::kInt64);

    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf* kernel_conf) const override {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    for (const std::string& obn : output_bns()) { BatchAxis4BnInOp(obn)->set_value(0); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLocalNonzeroConf, LocalNonzeroOp);

}  // namespace oneflow
