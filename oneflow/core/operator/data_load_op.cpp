#include "oneflow/core/operator/data_load_op.h"

namespace oneflow {

void DataLoadOp::InitFromOpConf() {
  CHECK(op_conf().has_data_load_conf());
  if (op_conf().data_load_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& DataLoadOp::GetCustomizedConf() const { return op_conf().data_load_conf(); }

void DataLoadOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  // TODO();
  // int64_t device_piece_size = GetBlobDesc4BnInOp("out")->shape().At(0);
  // kernel_conf->mutable_record_load_conf()->set_device_piece_size(device_piece_size);
}

void DataLoadOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                int64_t record_piece_size) const {
  // BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  // CHECK_EQ(record_piece_size % parallel_ctx->parallel_num(), 0);
  // out_blob_desc->mut_shape() = Shape({record_piece_size / parallel_ctx->parallel_num()});
  // out_blob_desc->set_data_type(kOFRecord);

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const data_load_conf& conf = op_conf().decode_random_conf();
  std::vector<int64_t> dim_vec(1 + conf.shape().dim_size());
  CHECK_EQ_OR_RETURN(record_piece_size % parallel_ctx->parallel_num(), 0);
  dim_vec[0] = record_piece_size / parallel_ctx->parallel_num();
  FOR_RANGE(size_t, j, 1, dim_vec.size()) { dim_vec[j] = conf.shape().dim(j - 1); }
  out_blob_desc->mut_shape() = Shape(dim_vec);
  out_blob_desc->set_data_type(conf.data_type());
  out_blob_desc->set_has_data_id_field(job_desc().SizeOfOneDataId() > 0);
  return Maybe<void>::Ok();
}

Maybe<void> DataLoadOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->set_value(0);
  return Maybe<void>::Ok();
}

Maybe<void> DataLoadOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kDataLoadConf, DataLoadOp);

}  // namespace oneflow
