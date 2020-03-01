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
};

void CategoricalHashTableLookupOp::InitFromOpConf() {
  CHECK(op_conf().has_categorical_hash_table_lookup_conf());
  if (!op_conf().categorical_hash_table_lookup_conf().hash_precomputed()) { UNIMPLEMENTED(); }
  EnrollInputBn("table", false)->set_is_mutable(true);
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
  const BlobDesc* table = GetBlobDesc4BnInOp("table");
  const DataType data_type = table->data_type();
  CHECK_OR_RETURN(IsIndexDataType(data_type));
  CHECK_EQ_OR_RETURN(table->shape().NumAxes(), 1);
  CHECK_EQ_OR_RETURN(table->shape().elem_cnt() % 2, 0);
  const BlobDesc* size = GetBlobDesc4BnInOp("size");
  CHECK_EQ_OR_RETURN(size->data_type(), data_type);
  CHECK_EQ_OR_RETURN(size->shape().NumAxes(), 1);
  CHECK_EQ_OR_RETURN(size->shape().elem_cnt(), 1);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in->data_type(), data_type);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  return Maybe<void>::Ok();
}

Maybe<void> CategoricalHashTableLookupOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_OR_RETURN(!BatchAxis4BnInOp("table")->has_value());
  CHECK_OR_RETURN(!BatchAxis4BnInOp("size")->has_value());
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kCategoricalHashTableLookupConf, CategoricalHashTableLookupOp);

}  // namespace oneflow
