#include "oneflow/core/operator/operator.h"

namespace oneflow {

class UserOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOp);
  UserOp() = default;
  ~UserOp() = default;

  void InitFromOpConf() override {
    // TODO();
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().user_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // TODO
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    // TODO
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    // TODO
    return Maybe<void>::Ok();
  }
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
    auto user_conf = kernel_conf->mutable_user_conf();
    *(user_conf->mutable_parallel_ctx()) = *parallel_ctx;
#define BLOB_DESCS_TO_PROTO(prefix)                         \
  for (const auto& bn : prefix##_bns()) {                   \
    BlobDescProto proto;                                    \
    GetBlobDesc4BnInOp(bn)->ToProto(&proto);                \
    (*user_conf->mutable_bn_in_op2blob_desc())[bn] = proto; \
  }

    BLOB_DESCS_TO_PROTO(input)
    BLOB_DESCS_TO_PROTO(output)
    BLOB_DESCS_TO_PROTO(tmp)

#undef BLOB_DESCS_TO_PROTO
  }
};

REGISTER_OP(OperatorConf::kUserConf, UserOp);

}  // namespace oneflow
