#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ExtractPieceSliceIdOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExtractPieceSliceIdOp);
  ExtractPieceSliceIdOp() = default;
  ~ExtractPieceSliceIdOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().extract_piece_slice_id_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*, const OpContext*) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    FOR_RANGE(int32_t, i, 0, output_bns().size()) {
      *BatchAxis4BnInOp(output_bns().Get(i)) = *BatchAxis4BnInOp(input_bns().Get(i));
    }
    return Maybe<void>::Ok();
  }
};

void ExtractPieceSliceIdOp::InitFromOpConf() {
  CHECK(op_conf().has_extract_piece_slice_id_conf());
  EnrollRepeatedInputBn("in", false);
  EnrollRepeatedOutputBn("out", false);
}

Maybe<void> ExtractPieceSliceIdOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().extract_piece_slice_id_conf();
  const DataType data_type = GetBlobDesc4BnInOp(input_bns().Get(0))->data_type();
  FOR_RANGE(int32_t, i, 1, conf.in_size()) {
    CHECK_EQ_OR_RETURN(data_type, GetBlobDesc4BnInOp(input_bns().Get(i))->data_type());
  }
  CHECK_EQ_OR_RETURN(conf.in_size(), conf.out_size());
  FOR_RANGE(int32_t, i, 0, conf.in_size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(input_bns().Get(i));
    BlobDesc* out_i = GetBlobDesc4BnInOp(output_bns().Get(i));
    out_i->mut_shape() = Shape({in_i->shape().At(0)});
    out_i->set_data_type(DataType::kInt32);
    out_i->set_is_dynamic(in_i->is_dynamic());
  }
  return Maybe<void>::Ok();
}

void ExtractPieceSliceIdOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp(input_bns().Get(0))->data_type());
}

REGISTER_OP(OperatorConf::kExtractPieceSliceIdConf, ExtractPieceSliceIdOp);

}  // namespace oneflow
