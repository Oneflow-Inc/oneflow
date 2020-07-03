#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

class BoxingIdentityOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingIdentityOp);
  BoxingIdentityOp() = default;
  ~BoxingIdentityOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 protected:
  virtual void VirtualInferBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}
  virtual void VirtualInitFromOpConf(){};

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

void BoxingIdentityOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalBlobId BoxingIdentityOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().boxing_identity_conf().lbi();
}

LogicalBlobId BoxingIdentityOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().boxing_identity_conf().lbi();
}

Maybe<void> BoxingIdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxingIdentityConf, BoxingIdentityOp);

}  // namespace oneflow
