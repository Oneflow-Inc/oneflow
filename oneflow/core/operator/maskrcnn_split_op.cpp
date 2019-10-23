#include "oneflow/core/operator/operator.h"

namespace oneflow {
class MaskrcnnSplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskrcnnSplitOp);
  MaskrcnnSplitOp() = default;
  ~MaskrcnnSplitOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_maskrcnn_split_conf());
    EnrollInputBn("in", false);
    EnrollRepeatedInputBn("segm", false);
    EnrollRepeatedOutputBn("out", false);
  }
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().maskrcnn_split_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const int32_t num_segm = op_conf().maskrcnn_split_conf().segm_size();
    CHECK_EQ_OR_RETURN(num_segm, op_conf().maskrcnn_split_conf().out_size());
    FOR_RANGE(int32_t, i, 0, num_segm) {
      const BlobDesc* segm_i = GetBlobDesc4BnInOp("segm_" + std::to_string(i));
      BlobDesc* out_i = GetBlobDesc4BnInOp("out_" + std::to_string(i));
      *out_i = *in;
      out_i->mut_shape().Set(0, segm_i->shape().At(0));
      out_i->set_is_dynamic(segm_i->is_dynamic());
    }

    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    for (const auto& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp("in"); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kMaskrcnnSplitConf, MaskrcnnSplitOp);

}  // namespace oneflow
