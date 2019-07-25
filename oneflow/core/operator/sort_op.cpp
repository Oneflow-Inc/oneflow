#include "oneflow/core/operator/sort_op.h"
#include "oneflow/core/kernel/radix_sort_util.h"

namespace oneflow {

void SortOp::InitFromOpConf() {
  CHECK(op_conf().has_sort_conf());
  EnrollInputBn("key", false);
  EnrollInputBn("value", false);
  EnrollDataTmpBn("temp_storage");
  EnrollOutputBn("sorted_key", false);
  EnrollOutputBn("sorted_value", false);
}

const PbMessage& SortOp::GetCustomizedConf() const { return this->op_conf().sort_conf(); }

void SortOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, int64_t record_piece_size,
                            std::function<void(OpContext*)> EnrollOpCtx) const {
  // input
  const BlobDesc* key = GetBlobDesc4BnInOp("key");
  const BlobDesc* value = GetBlobDesc4BnInOp("value");
  CHECK_EQ(key->shape(), value->shape());
  CHECK_EQ(value->data_type(), DataType::kInt32);

  // data_tmp: temp_storage
  int64_t temp_storage_bytes =
      InferTempStorageForRadixSort(key->shape().At(0), key->shape().At(1), key->data_type());
  BlobDesc* temp_storage = GetBlobDesc4BnInOp("temp_storage");
  temp_storage->set_data_type(DataType::kChar);
  temp_storage->mut_shape() = Shape({temp_storage_bytes});
  SortOpCtx* sort_op_ctx = new SortOpCtx(temp_storage_bytes);
  EnrollOpCtx(sort_op_ctx);

  // output
  *GetBlobDesc4BnInOp("sorted_key") = *key;
  *GetBlobDesc4BnInOp("sorted_value") = *value;
}

void SortOp::VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)>,
                                  const ParallelContext*, KernelConf* kernel_conf,
                                  const OpContext* op_ctx) const {
  auto* sort_op_ctx = static_cast<const SortOpCtx*>(op_ctx);
  kernel_conf->mutable_sort_conf()->set_temp_storage_bytes(sort_op_ctx->GetTempStorageBytes());
}

REGISTER_OP(OperatorConf::kSortConf, SortOp);

}  // namespace oneflow
