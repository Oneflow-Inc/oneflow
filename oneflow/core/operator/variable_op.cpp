#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

// S(0) -> C
std::unique_ptr<const OpParallelSignature> MakeVariableOpDataSplitOpParallelSignature(
    const Operator* op) {
  std::string desc = op->op_name() + ": S(0) -> C";
  auto IsMatched = [op](const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
                        const ParallelContext* parallel_ctx) {
    OpParallelMatchResult default_ret;
    if (parallel_ctx->policy() == kDataParallel) {
      return MakeOpParallelMatchSuccess();
    } else {
      return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kDataParallel);
    }
  };
  auto GenSignature = [op](
                          const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
                          HashMap<std::string, LogicalBlobParallelDesc>* signature) {
    CHECK(LbpdHint4BnInOp("tick").has_data_split());
    CHECK(LbpdHint4BnInOp("out").has_model_clone());
    (*signature)["tick"].mutable_split_parallel()->set_axis(0);
    (*signature)["out"].mutable_broadcast_parallel();
  };
  return std::make_unique<OpParallelSignature>(desc, IsMatched, GenSignature);
}

// S(0) -> S
std::unique_ptr<const OpParallelSignature> MakeVariableOpModelSplitOpParallelSignature(
    const Operator* op) {
  std::string desc = op->op_name() + ": S(0) -> S";
  auto IsMatched = [op](const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
                        const ParallelContext* parallel_ctx) {
    OpParallelMatchResult default_ret;
    if (parallel_ctx->policy() == kModelParallel) {
      return MakeOpParallelMatchSuccess();
    } else {
      return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
    }
  };
  auto GenSignature = [op](
                          const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
                          HashMap<std::string, LogicalBlobParallelDesc>* signature) {
    CHECK(LbpdHint4BnInOp("tick").has_data_split());
    CHECK(LbpdHint4BnInOp("out").has_model_split());
    int32_t axis = LbpdHint4BnInOp("out").model_split().axis();
    (*signature)["tick"].mutable_split_parallel()->set_axis(0);
    (*signature)["out"].mutable_split_parallel()->set_axis(axis);
  };
  return std::make_unique<OpParallelSignature>(desc, IsMatched, GenSignature);
}

}  // namespace

void VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  EnrollInputBn("tick", false);
  EnrollOutputBn("out", Global<JobDesc>::Get()->IsTrain() && op_conf().trainable());
  EnrollModelBn(op_conf().variable_conf().model_name());
}

const PbMessage& VariableOp::GetCustomizedConf() const { return op_conf().variable_conf(); }

int32_t VariableOp::ModelSplitAxis() const { return op_conf().variable_conf().model_split_axis(); }

void VariableOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  BlobDesc* model_blob_desc = GetBlobDesc4BnInOp(variable_conf.model_name());
  model_blob_desc->mut_shape() = Shape(variable_conf.shape());
  model_blob_desc->set_data_type(variable_conf.has_data_type()
                                     ? variable_conf.data_type()
                                     : Global<JobDesc>::Get()->DefaultDataType());
  if (parallel_ctx->policy() == kModelParallel) {
    int32_t model_split_axis = variable_conf.model_split_axis();
    CHECK_GE(model_split_axis, 0);
    CHECK_LT(model_split_axis, model_blob_desc->shape().NumAxes());
    int64_t split_dim_num = model_blob_desc->shape().At(model_split_axis);
    BalancedSplitter bs(split_dim_num, parallel_ctx->parallel_num());
    model_blob_desc->mut_shape().Set(model_split_axis, bs.At(parallel_ctx->parallel_id()).size());
  } else {
    CHECK_EQ(parallel_ctx->policy(), kDataParallel);
  }
  *GetBlobDesc4BnInOp("out") = *model_blob_desc;
}

void VariableOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(MakeVariableOpDataSplitOpParallelSignature(this));
  op_parallel_signatures->emplace_back(MakeVariableOpModelSplitOpParallelSignature(this));
}

void VariableOp::InferOutputBlobLbpdHint(
    std::function<LbpdHint*(const std::string&)> LbpdHint4BnInOp,
    std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
    const ParallelContext* parallel_context) const {
  if (parallel_context->policy() == kDataParallel) {
    LbpdHint4BnInOp("out")->mutable_model_clone();
  } else if (parallel_context->policy() == kModelParallel) {
    LbpdHint4BnInOp("out")->mutable_model_split()->set_axis(ModelSplitAxis());
  } else {
    UNIMPLEMENTED();
  }
}

void VariableOp::InferIsModelBlob4OutputBlobs(
    std::function<bool*(const std::string&)> IsModelBlob4BnInOp) const {
  *IsModelBlob4BnInOp("out") = true;
}

void VariableOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* conf) const {
  conf->mutable_variable_conf()->set_is_fw_inplace(*is_fw_inplace_);
  conf->mutable_variable_conf()->set_is_bw_inplace(*is_bw_inplace_);
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);

}  // namespace oneflow
