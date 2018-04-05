#include "oneflow/core/operator/normalization_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

bool NormalizationOp::HasScaleOrCenter() const {
  const auto& conf = op_conf().normalization_conf();
  return conf.center() || conf.scale();
}

void NormalizationOp::InitFromOpConf() {
  const auto& conf = op_conf().normalization_conf();
  CHECK_GT(conf.epsilon(), 0.f);
  CHECK_GE(conf.momentum(), 0);
  CHECK_LE(conf.momentum(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("new_mean");
  EnrollDataTmpBn("new_variance");
  EnrollForwardModelBn("moving_mean");
  EnrollForwardModelBn("moving_variance");

  if (conf.center()) {
    EnrollModelBn("beta");
  } else {
    EnrollDataTmpBn("beta_diff");
  }
  if (conf.scale()) {
    EnrollModelBn("gamma");
  } else {
    EnrollDataTmpBn("gamma_diff");
  }
  EnrollDataTmpBn("normalized_in");
  EnrollDataTmpBn("inv_var");
  EnrollDataTmpBn("tmp_storage_for_sum");
  EnrollDataTmpBn("trans_in");
  EnrollDataTmpBn("trans_out");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const auto& conf = op_conf().normalization_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const DataType in_data_type = in_blob_desc->data_type();
  CHECK_EQ(in_data_type, Global<JobDesc>::Get()->DefaultDataType());
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  NormalizationOpCtx* op_ctx = NewNormalizationOpCtx(in_blob_desc->shape());
  EnrollOpCtx(op_ctx);
  if (op_ctx->need_transpose) {
    BlobDesc* transpose_blob_desc = GetBlobDesc4BnInOp("trans_in");
    transpose_blob_desc->mut_shape() = in_blob_desc->shape();
    transpose_blob_desc->mut_shape().Set(op_ctx->axis,
                                         in_blob_desc->shape().At(0));
    transpose_blob_desc->mut_shape().Set(0, op_ctx->transpose_cols);
    transpose_blob_desc->set_data_type(in_data_type);
    *GetBlobDesc4BnInOp("trans_out") = *transpose_blob_desc;
    *GetBlobDesc4BnInOp("normalized_in") = *transpose_blob_desc;
  } else {
    *GetBlobDesc4BnInOp("normalized_in") = *in_blob_desc;
  }

  BlobDesc blob_desc(Shape({op_ctx->transpose_cols}), in_data_type, false,
                     false, 1);
  std::list<std::string> blob_names = {"moving_mean", "moving_variance",
                                       "inv_var"};
  std::list<std::string> bns_needless_in_predict = {"new_mean", "new_variance"};
  if (conf.center()) {
    blob_names.push_back("beta");
  } else {
    blob_names.push_back("beta_diff");
  }
  if (conf.scale()) {
    blob_names.push_back("gamma");
  } else {
    blob_names.push_back("gamma_diff");
  }
  if (Global<JobDesc>::Get()->IsTrain()) {
    for (const std::string& bn : bns_needless_in_predict) {
      blob_names.push_back(bn);
    }
  }
  for (const auto& bn_in_op : blob_names) {
    *GetBlobDesc4BnInOp(bn_in_op) = blob_desc;
  }
  size_t tmp_storage_size = 0;
  if (device_type == DeviceType::kGPU) {
    tmp_storage_size =
        GetTmpSizeForReduceSum(in_data_type, op_ctx->transpose_rows);
    CHECK_GT(tmp_storage_size, 0);
  }
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("tmp_storage_for_sum");
  tmp_blob_desc->set_data_type(in_data_type);
  tmp_blob_desc->mut_shape() = Shape(
      {static_cast<int64_t>(tmp_storage_size / GetSizeOfDataType(in_data_type))
       + 1});
}

void NormalizationOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf,
    const OpContext* op_ctx) const {
  NormalizationKernelConf* conf = kernel_conf->mutable_normalization_conf();
  const NormalizationOpCtx* ctx =
      static_cast<const NormalizationOpCtx*>(op_ctx);
  conf->set_axis(ctx->axis);
  conf->set_transpose_rows(ctx->transpose_rows);
  conf->set_transpose_cols(ctx->transpose_cols);
  conf->set_need_transpose(ctx->need_transpose);
  if (ctx->need_transpose) {
    PbRf<int32_t>* perm = conf->mutable_perm();
    perm->Reserve(ctx->dims);
    for (size_t i = 0; i < ctx->dims; ++i) { perm->Add(i); }
    perm->SwapElements(ctx->axis, 0);
  }
}

void NormalizationOp::VirtualFixParallelDesc(ParallelDesc* pr_desc) const {
  pr_desc->set_policy(ParallelPolicy::kDataParallel);
}

NormalizationOpCtx* NormalizationOp::NewNormalizationOpCtx(
    const Shape& in_shape) const {
  NormalizationOpCtx* op_ctx = new NormalizationOpCtx();
  op_ctx->axis = op_conf().normalization_conf().axis();
  op_ctx->dims = in_shape.NumAxes();
  if (op_ctx->axis < 0) { op_ctx->axis += op_ctx->dims; }
  CHECK_GE(op_ctx->axis, 0);
  CHECK_LT(op_ctx->axis, op_ctx->dims);
  op_ctx->transpose_cols = in_shape.At(op_ctx->axis);
  op_ctx->transpose_rows = in_shape.elem_cnt() / op_ctx->transpose_cols;
  if (op_ctx->axis == 0) {
    op_ctx->need_transpose = false;
  } else {
    op_ctx->need_transpose = true;
  }
  return op_ctx;
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
