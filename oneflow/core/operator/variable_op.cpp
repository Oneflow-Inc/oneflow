#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

// S(0) -> C
const OpParallelSignature MakeVariableOpDataSplitParallelSignature(const Operator* op) {
  std::string desc = op->op_name() + ": S(0) -> C";
  auto IsMatched =
      [op](
          const std::function<const LogicalBlobParallelDesc&(const std::string&)>& ProducerLbpd4Ibn,
          const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
          const ParallelContext* parallel_ctx) {
        OpParallelMatchResult default_ret;
        if (parallel_ctx->policy() == kDataParallel) {
          return MakeOpParallelMatchSuccess();
        } else {
          return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kDataParallel);
        }
      };
  auto GenSignature = [op](const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
                           HashMap<std::string, LogicalBlobParallelDesc>* signature) {
    CHECK_EQ(ModelSplitAxis4BnInOp("tick"), -1);
    CHECK_EQ(ModelSplitAxis4BnInOp("out"), -1);
    (*signature)["tick"].mutable_split_parallel()->set_axis(0);
    (*signature)["out"].mutable_clone_parallel();
  };
  return OpParallelSignature(desc, IsMatched, GenSignature);
}

// S(0) -> S
const OpParallelSignature MakeVariableOpModelSplitParallelSignature(const Operator* op) {
  std::string desc = op->op_name() + ": S(0) -> S";
  auto IsMatched =
      [op](
          const std::function<const LogicalBlobParallelDesc&(const std::string&)>& ProducerLbpd4Ibn,
          const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
          const ParallelContext* parallel_ctx) {
        OpParallelMatchResult default_ret;
        if (parallel_ctx->policy() == kModelParallel) {
          return MakeOpParallelMatchSuccess();
        } else {
          return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
        }
      };
  auto GenSignature = [op](const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
                           HashMap<std::string, LogicalBlobParallelDesc>* signature) {
    CHECK_EQ(ModelSplitAxis4BnInOp("tick"), -1);
    CHECK_NE(ModelSplitAxis4BnInOp("out"), -1);
    (*signature)["tick"].mutable_split_parallel()->set_axis(0);
    (*signature)["out"].mutable_split_parallel()->set_axis(ModelSplitAxis4BnInOp("out"));
  };
  return OpParallelSignature(desc, IsMatched, GenSignature);
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

/*
void VariableOp::InferInputOutputLogicalBlobParallelDesc(
    std::function<LogicalBlobParallelDesc*(const std::string&)> LogicalBlobParallelDesc4BnInOp,
    std::function<const LogicalBlobParallelDesc&(const std::string&)>
ProducerLogicalBlobParallelDesc4Ibn, std::function<int32_t(const std::string&)>
ModelSplitAxis4BnInOp, const ParallelContext* parallel_ctx) const {
  LogicalBlobParallelDesc4BnInOp("tick")->mutable_clone_parallel();
  if (parallel_ctx->policy() == kDataParallel) {
    LogicalBlobParallelDesc4BnInOp("out")->mutable_clone_parallel();
  } else if (parallel_ctx->policy() == kModelParallel) {
    LogicalBlobParallelDesc4BnInOp("out")->mutable_split_parallel(ModelSplitAxis4BnInOp("out"));
  } else {
    UNIMPLEMENTED();
  }
}
*/

void VariableOp::InitOpParallelSignatures() {
  mut_op_parallel_signatures()->push_back(MakeVariableOpDataSplitParallelSignature(this));
  mut_op_parallel_signatures()->push_back(MakeVariableOpModelSplitParallelSignature(this));
}

void VariableOp::InferOutputBlobParallelDesc(
    std::function<BlobParallelDesc*(const std::string&)> BlobParallelDesc4BnInOp,
    const ParallelContext* parallel_context) const {
  ModelBlobParallel* model_blob_parallel =
      BlobParallelDesc4BnInOp("out")->mut_model_blob_parallel();
  if (parallel_context->policy() == kDataParallel) {
    model_blob_parallel->set_clone_num(parallel_context->parallel_num());
    model_blob_parallel->set_model_split_num(1);
  } else if (parallel_context->policy() == kModelParallel) {
    model_blob_parallel->set_clone_num(1);
    model_blob_parallel->set_model_split_num(parallel_context->parallel_num());
  } else {
    UNIMPLEMENTED();
  }
}

void VariableOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* conf) const {
  conf->mutable_variable_conf()->set_is_fw_inplace(*is_fw_inplace_);
  conf->mutable_variable_conf()->set_is_bw_inplace(*is_bw_inplace_);
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);

}  // namespace oneflow
