#include "oneflow/core/operator/constant_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ConstantOp::InitFromOpConf() {
  CHECK(op_conf().has_constant_conf());
  if (op_conf().constant_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& ConstantOp::GetCustomizedConf() const { return op_conf().constant_conf(); }

Maybe<void> ConstantOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->policy(), ParallelPolicy::kDataParallel);
  const ConstantOpConf& conf = op_conf().constant_conf();
  const DataType& data_type =
      conf.has_data_type() ? conf.data_type() : GlobalJobDesc().DefaultDataType();
  std::vector<int64_t> dim_vec;
  if (conf.use_device_piece_size_as_dim0()) {
    CHECK_EQ_OR_RETURN(record_piece_size % parallel_ctx->parallel_num(), 0);
    dim_vec.push_back(record_piece_size / parallel_ctx->parallel_num());
  }
  if (conf.has_shape()) {
    dim_vec.insert(dim_vec.end(), conf.shape().dim().cbegin(), conf.shape().dim().cend());
  }
  if (dim_vec.empty()) { dim_vec.push_back(1); }
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(data_type);
  out->mut_shape() = Shape(dim_vec);
  return Maybe<void>::Ok();
}

void ConstantOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_constant_conf()->set_random_seed(NewRandomSeed());
  const DataType& data_type = GetBlobDesc4BnInOp("out")->data_type();
  if (op_conf().constant_conf().has_initializer()) {
    *kernel_conf->mutable_constant_conf()->mutable_initializer() =
        op_conf().constant_conf().initializer();
  } else if (IsFloatingDataType(data_type)) {
    InitializerConf conf;
    conf.mutable_constant_conf()->set_value(0);
    *kernel_conf->mutable_constant_conf()->mutable_initializer() = conf;
  } else if (IsIntegralDataType(data_type)) {
    InitializerConf conf;
    conf.mutable_constant_int_conf()->set_value(0);
    *kernel_conf->mutable_constant_conf()->mutable_initializer() = conf;
  } else {
    UNIMPLEMENTED();
  }
  kernel_conf->set_data_type(data_type);
}

Maybe<void> ConstantOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

void ConstantOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kConstantConf, ConstantOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kConstantConf, 1);

}  // namespace oneflow
