#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class CategoricalHashTableLookupOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CategoricalHashTableLookupOp);
  CategoricalHashTableLookupOp() = default;
  ~CategoricalHashTableLookupOp() override = default;

 private:
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

void CategoricalHashTableLookupOp::InitFromOpConf() {
  CHECK(op_conf().has_categorical_hash_table_lookup_conf());
  if (!op_conf().categorical_hash_table_lookup_conf().hash_precomputed()) { UNIMPLEMENTED(); }
  EnrollInputBn("key", false)->set_is_mutable(true);
  EnrollInputBn("value", false)->set_is_mutable(true);
  EnrollInputBn("size", false)->set_is_mutable(true);
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& CategoricalHashTableLookupOp::GetCustomizedConf() const {
  return op_conf().categorical_hash_table_lookup_conf();
}

Maybe<void> CategoricalHashTableLookupOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  const BlobDesc* key = GetBlobDesc4BnInOp("key");
  const DataType key_data_type = key->data_type();
  CHECK_OR_RETURN(IsIndexDataType(key_data_type));
  CHECK_EQ_OR_RETURN(key->shape().NumAxes(), 1);
  const BlobDesc* value = GetBlobDesc4BnInOp("value");
  const DataType value_data_type = value->data_type();
  CHECK_OR_RETURN(IsIndexDataType(value_data_type));
  CHECK_EQ_OR_RETURN(value->shape().NumAxes(), 1);
  CHECK_EQ_OR_RETURN(value->shape().elem_cnt(), key->shape().elem_cnt());
  const BlobDesc* size = GetBlobDesc4BnInOp("size");
  CHECK_EQ_OR_RETURN(size->data_type(), value_data_type);
  CHECK_EQ_OR_RETURN(size->shape().NumAxes(), 1);
  CHECK_EQ_OR_RETURN(size->shape().elem_cnt(), 1);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in->data_type(), key_data_type);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->set_data_type(value_data_type);
  return Maybe<void>::Ok();
}

Maybe<void> CategoricalHashTableLookupOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_OR_RETURN(!BatchAxis4BnInOp("key")->has_value());
  CHECK_OR_RETURN(!BatchAxis4BnInOp("value")->has_value());
  CHECK_OR_RETURN(!BatchAxis4BnInOp("size")->has_value());
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
  return Maybe<void>::Ok();
}

void CategoricalHashTableLookupOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("value")->data_type());
  kernel_conf->mutable_categorical_hash_table_lookup_conf()->set_key_data_type(
      GetBlobDesc4BnInOp("key")->data_type());
}

REGISTER_OP(OperatorConf::kCategoricalHashTableLookupConf, CategoricalHashTableLookupOp);

}  // namespace oneflow
