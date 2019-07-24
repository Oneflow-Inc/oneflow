#include "oneflow/core/operator/sort_op.h"

namespace oneflow {

void SortOp::InitFromOpConf() {
  CHECK(op_conf().has_sort_conf());
  EnrollInputBn("key", false);
  EnrollInputBn("value", false);
  EnrollOutputBn("sorted_key", false);
  EnrollOutputBn("sorted_value", false);

  EnrollDataTmpBn("temp_storage");
}

const PbMessage& SortOp::GetCustomizedConf() const { return this->op_conf().sort_conf(); }

void SortOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  // input
  const BlobDesc* key = GetBlobDesc4BnInOp("key");
  const BlobDesc* value = GetBlobDesc4BnInOp("value");
  CHECK_EQ(key->shape(), value->shape());
  CHECK_EQ(value->data_type(), DataType::kInt32);

  // output
  *GetBlobDesc4BnInOp("sorted_key") = *key;
  *GetBlobDesc4BnInOp("sorted_value") = *value;

  // data_tmp: temp_storage
  BlobDesc* temp_storage = GetBlobDesc4BnInOp("temp_storage");
  temp_storage->set_data_type(DataType::kChar);
  temp_storage->mut_shape() = Shape({op_conf().sort_conf().temp_storage_bytes()});
}

REGISTER_OP(OperatorConf::kSortConf, SortOp);

}  // namespace oneflow
