#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class UniqueWithCountsOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniqueWithCountsOp);
  UniqueWithCountsOp() = default;
  ~UniqueWithCountsOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

void UniqueWithCountsOp::InitFromOpConf() {
  CHECK(op_conf().has_unique_with_counts_conf());
  EnrollInputBn("x", false);
  EnrollOutputBn("y", false);
  EnrollOutputBn("idx", false);
  EnrollOutputBn("count", false);
  EnrollOutputBn("num_unique", false);
}

const PbMessage& UniqueWithCountsOp::GetCustomizedConf() const { return op_conf().unique_with_counts_conf(); }

Maybe<void> UniqueWithCountsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  CHECK_EQ_OR_RETURN(x->shape().NumAxes(), 1);
  CHECK(IsIndexDataType(x->data_type()));
  BlobDesc* y = GetBlobDesc4BnInOp("y");
  *y = *x;
  const DataType idx_data_type = op_conf().unique_with_counts_conf().out_idx();
  BlobDesc* idx = GetBlobDesc4BnInOp("idx");
  *idx = *x;
  idx->set_data_type(idx_data_type);
  BlobDesc* count = GetBlobDesc4BnInOp("count");
  *count = *x;
  count->set_data_type(idx_data_type);
  BlobDesc* num_unique = GetBlobDesc4BnInOp("num_unique");
  num_unique->mut_shape() = Shape({1});
  num_unique->set_data_type(idx_data_type);
  return Maybe<void>::Ok();
}

void UniqueWithCountsOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
}

REGISTER_OP(OperatorConf::kUniqueWithCountsConf, UniqueWithCountsOp);

}  // namespace oneflow
