#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

int64_t InferKeyValueOutByteSize(const Shape& shape, DataType data_type) {
  int32_t bytes = 0;
  switch (data_type) {
    case DataType::kFloat: bytes = sizeof(float); break;
    case DataType::kDouble: bytes = sizeof(double); break;
    case DataType::kInt32: bytes = sizeof(int32_t); break;
    case DataType::kInt8: bytes = sizeof(int8_t); break;
    default: UNIMPLEMENTED(); break;
  }
  return shape.Count(0, shape.NumAxes() - 1) * (bytes + sizeof(int32_t));
}

}  // namespace

class ArgmaxOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgmaxOp);
  ArgmaxOp() = default;
  ~ArgmaxOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_argmax_conf());
    EnrollInputBn("in", false);
    EnrollTmpBn("temp_storage");
    EnrollTmpBn("key_value_out");
    EnrollOutputBn("out", false);
  }
  const PbMessage& GetCustomizedConf() const override { return this->op_conf().argmax_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    // fw_buf: temp_storage
    int64_t temp_storage_bytes = 100;
    BlobDesc* temp_storage = GetBlobDesc4BnInOp("temp_storage");
    temp_storage->mut_shape() = Shape({temp_storage_bytes});
    temp_storage->set_data_type(DataType::kChar);
    temp_storage->set_is_dynamic(false);
    // fw_buf: key_value_out
    BlobDesc* key_value_out = GetBlobDesc4BnInOp("key_value_out");
    key_value_out->mut_shape() = Shape({InferKeyValueOutByteSize(in->shape(), in->data_type())});
    key_value_out->set_data_type(DataType::kChar);
    key_value_out->set_is_dynamic(in->is_dynamic());
    // output
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    auto dim_vec = in->shape().dim_vec();
    dim_vec.pop_back();
    out->mut_shape() = Shape(dim_vec);
    out->set_data_type(DataType::kInt32);
    out->set_is_dynamic(in->is_dynamic());

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
  }
};

REGISTER_OP(OperatorConf::kArgmaxConf, ArgmaxOp);

}  // namespace oneflow
