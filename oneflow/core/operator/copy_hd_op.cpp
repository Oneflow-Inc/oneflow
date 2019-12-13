#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CopyHdOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdOp);
  CopyHdOp() = default;
  ~CopyHdOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

void CopyHdOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

Maybe<void> CopyHdOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

const PbMessage& CopyHdOp::GetCustomizedConf() const { return op_conf().copy_hd_conf(); }

LogicalBlobId CopyHdOp::ibn2lbi(const std::string& input_bn) const {
  if (this->op_conf().copy_hd_conf().has_lbi()) {
    return this->op_conf().copy_hd_conf().lbi();
  } else {
    return GenPackedLbi();
  }
}

LogicalBlobId CopyHdOp::obn2lbi(const std::string& output_bn) const {
  if (this->op_conf().copy_hd_conf().has_lbi()) {
    return this->op_conf().copy_hd_conf().lbi();
  } else {
    return GenPackedLbi();
  }
}

REGISTER_OP(OperatorConf::kCopyHdConf, CopyHdOp);

}  // namespace oneflow
