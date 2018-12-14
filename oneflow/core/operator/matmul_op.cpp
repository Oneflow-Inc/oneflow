#include "oneflow/core/operator/matmul_op.h"
#include "oneflow/core/common/balanced_splitter.h"
namespace oneflow {

void MatmulOp::InitFromOpConf() {
  CHECK(op_conf().has_matmul_conf());

  EnrollInputBn("in");
  EnrollInputBn("weight");
  EnrollOutputBn("out");
  if (op_conf().matmul_conf().has_bias()) {
    EnrollInputBn("bias");
    EnrollConstBufBn("bias_multiplier");
  }
}

const PbMessage& MatmulOp::GetCustomizedConf() const { return op_conf().matmul_conf(); }

void MatmulOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const MatmulOpConf& conf = op_conf().matmul_conf();
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  int32_t units = conf.units();
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(units, parallel_ctx->parallel_num());
    units = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), units});
  out_blob_desc->set_has_instance_shape_field(false);
  if (conf.has_bias()) {
    // bias_multiplier
    GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() = Shape({in_blob_desc->shape().At(0), 1});
  }
}

REGISTER_OP(OperatorConf::kMatmulConf, MatmulOp);

}  // namespace oneflow
