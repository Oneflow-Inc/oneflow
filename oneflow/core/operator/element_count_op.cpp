#include "oneflow/core/operator/operator.h"

namespace oneflow {
class ElementCountOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementCountOp);
  ElementCountOp() = default;
  ~ElementCountOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().element_count_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    BatchAxis4BnInOp("out")->clear_value();
    return Maybe<void>::Ok();
  }
};
void ElementCountOp::InitFromOpConf() {
  CHECK(op_conf().has_element_count_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

Maybe<void> ElementCountOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ElementCountOpConf& conf = op_conf().element_count_conf();
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({1});
  const DataType data_type =
      conf.has_data_type() ? conf.data_type() : Global<JobDesc>::Get()->DefaultDataType();
  out->set_data_type(data_type);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kElementCountConf, ElementCountOp);

}  // namespace oneflow
